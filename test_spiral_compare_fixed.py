#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, os, time
from pymavlink import mavutil

M_PER_DEG = 111319.49079327357
STATES  = ["L", "M", "R"]
ACTIONS = ["D", "U", "I"]

def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_360(deg):
    deg = deg % 360.0
    return deg if deg >= 0 else deg + 360.0

def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def latlon_to_ne(lat_deg, lon_deg, lat0_deg, lon0_deg):
    dlat = (lat_deg - lat0_deg)
    dlon = (lon_deg - lon0_deg)
    north = dlat * M_PER_DEG
    east  = dlon * M_PER_DEG * math.cos(math.radians(lat0_deg))
    return north, east

def ne_to_latlon(north_m, east_m, lat0_deg, lon0_deg):
    dlat = north_m / M_PER_DEG
    dlon = east_m / (M_PER_DEG * math.cos(math.radians(lat0_deg)))
    return lat0_deg + dlat, lon0_deg + dlon

def shift_latlon(lat_deg, lon_deg, north_m, east_m):
    return ne_to_latlon(north_m, east_m, lat_deg, lon_deg)

def ne_error_from_latlon(lat_deg, lon_deg, tgt_lat_deg, tgt_lon_deg):
    lat_rad = math.radians(lat_deg)
    dlat = (tgt_lat_deg - lat_deg)
    dlon = (tgt_lon_deg - lon_deg)
    north = dlat * M_PER_DEG
    east  = dlon * M_PER_DEG * math.cos(lat_rad)
    return north, east

def cross_track_error(n_err, e_err, tgt_heading_deg):
    chi = math.radians(wrap_360(tgt_heading_deg))
    right_n = -math.sin(chi)
    right_e =  math.cos(chi)
    return n_err * right_n + e_err * right_e

def state_from_dr(dr_m, deadband):
    if dr_m < -deadband: return "L"
    if dr_m >  deadband: return "R"
    return "M"

def reward_from_error(err_abs, e_expected, r_pos=100, r_neg=-1):
    return r_pos if err_abs <= e_expected else r_neg

def greedy_action(Q, s):
    row = Q.get(s, {})
    vals = {a: float(row.get(a, 0.0)) for a in ACTIONS}
    best_val = max(vals.values())
    best = [a for a, v in vals.items() if v == best_val]
    if "U" in best: return "U"
    return best[0]

def load_qtable(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    Q = obj.get("Q", None)
    if Q is None:
        raise ValueError("qtable json has no 'Q' field.")
    for s in STATES:
        Q.setdefault(s, {})
        for a in ACTIONS:
            Q[s].setdefault(a, 0.0)
    return Q

# MAVLink helpers
def send_guided_goto(master, lat_deg, lon_deg, alt_rel_m):
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    master.mav.mission_item_int_send(
        master.target_system, master.target_component,
        0, frame,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        2, 0,
        0, 0, 0, 0,
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        float(alt_rel_m)
    )

def send_guided_change_speed(master, groundspeed_mps):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43000, 0, 0,
        1, float(groundspeed_mps), 0.0, 0,
        0, 0, 0
    )

def send_guided_change_heading(master, heading_deg, max_centrip_acc=2.0):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43002, 0, 0,
        0, float(heading_deg), float(max_centrip_acc), 0,
        0, 0, 0
    )

def send_guided_change_alt(master, target_alt_m, rate_mps=2.0):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43001, 0, 0,
        0, 0, float(rate_mps), 0,
        0, 0, float(target_alt_m)
    )

def set_param(master, name: str, value: float):
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def mean(xs): return sum(xs) / max(1, len(xs))
def rms(xs):  return math.sqrt(sum([x*x for x in xs]) / max(1, len(xs)))

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--master", default="udp:127.0.0.1:14550")

    # ✅ 选择算法：baseline固定K / ql查Q表
    ap.add_argument("--algo", choices=["baseline", "ql"], default="ql",
                    help="baseline=固定增益K；ql=Q-learning查表调K")

    # ql 需要qtable；baseline不需要（但可以给，不用也行）
    ap.add_argument("--qtable", default="", help="Q-learning算法的q表json（algo=ql时必填）")

    ap.add_argument("--out_csv", default="test_log.csv")
    ap.add_argument("--out_summary", default="test_summary.json")

    # Spiral: 每圈+80
    ap.add_argument("--turns", type=float, default=3.0)
    ap.add_argument("--r0", type=float, default=80.0)
    ap.add_argument("--r_step", type=float, default=80.0)
    ap.add_argument("--direction", choices=["CW","CCW"], default="CW")

    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--cmd_hz", type=float, default=5.0)
    ap.add_argument("--centrip_acc", type=float, default=2.0)

    # Phase
    ap.add_argument("--wind_enable_alt", type=float, default=20.0)
    ap.add_argument("--spiral_start_alt", type=float, default=100.0)
    ap.add_argument("--spiral_climb", type=float, default=200.0)
    ap.add_argument("--climb_rate", type=float, default=2.0)

    # Wind
    ap.add_argument("--wind_from_deg", type=float, default=180.0)
    ap.add_argument("--wind_speed_mps", type=float, default=0.0)
    ap.add_argument("--wind_turb", type=float, default=0.0)
    ap.add_argument("--wind_ramp_s", type=float, default=0.0)

    # Eval thresholds (for reporting)
    ap.add_argument("--deadband", type=float, default=5.0)
    ap.add_argument("--e_expected", type=float, default=5.0)

    # Gain K
    ap.add_argument("--K_fixed", type=float, default=0.8, help="baseline固定K值")
    ap.add_argument("--K_init", type=float, default=1.0, help="ql初始K")
    ap.add_argument("--K_step", type=float, default=0.10)
    ap.add_argument("--K_min", type=float, default=0.5)
    ap.add_argument("--K_max", type=float, default=2.0)

    # Control
    ap.add_argument("--L_corr", type=float, default=50.0)
    ap.add_argument("--max_corr_deg", type=float, default=25.0)
    ap.add_argument("--lookahead_deg", type=float, default=15.0)

    # Center offset so start point lies on r0 circle
    ap.add_argument("--start_radial_deg", type=float, default=90.0)

    args = ap.parse_args()

    Q = None
    if args.algo == "ql":
        if not args.qtable:
            raise SystemExit("algo=ql requires --qtable path")
        Q = load_qtable(args.qtable)
        print("[QTABLE] loaded:", args.qtable, "policy:", {s: greedy_action(Q, s) for s in STATES})
    else:
        print("[BASELINE] fixed K =", args.K_fixed)

    master = mavutil.mavlink_connection(args.master)
    master.wait_heartbeat()
    print("Heartbeat OK")

    # reset wind residual
    set_param(master, "SIM_WIND_SPD", 0.0)
    set_param(master, "SIM_WIND_TURB", 0.0)

    wind_enabled = False
    wind_speed_now = 0.0
    wind_enable_time = None

    spiral_started = False
    center_lat = center_lon = None
    dir_sign = -1.0 if args.direction == "CW" else 1.0
    h_start = None

    phi_prev = None
    phi_accum = 0.0
    turns_completed_monotone = 0.0
    total_turns = float(args.turns)

    K = args.K_fixed if args.algo == "baseline" else args.K_init

    cmd_dt = 1.0 / args.cmd_hz
    last_step = time.time()

    last_gp = None
    
    last_att = None  # ATTITUDE: roll/pitch/yaw (rad)
    last_nav = None  # (optional) NAV_CONTROLLER_OUTPUT

    abs_dr, dr_list = [], []
    within5, n_metric, max_abs = 0, 0, 0.0

    t0 = time.time()
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s","algo","K","state","action",
            "dr_m","abs_dr_m","reward",
            "lat_deg","lon_deg","rel_alt_m",
            "ref_lat_deg","ref_lon_deg","ref_alt_m","ref_heading_deg",
            "tgt_lat_deg","tgt_lon_deg","tgt_alt_m",
            "tgt_heading_deg","cmd_heading_deg",
            "r_ref_m","r_now_m","turns_completed","progress",
            "wind_speed_mps","wind_from_deg",
            "roll_deg","pitch_deg","yaw_deg"

        ])

        print("Running... Ctrl+C to stop.")
        done = False
        try:
            while not done:
                #m = master.recv_match(type=["GLOBAL_POSITION_INT"], blocking=False)
                #if m and m.get_type() == "GLOBAL_POSITION_INT":
                    #last_gp = m
                 
                # 尽量把缓存更新到“最新”
                for _ in range(20):
                    m = master.recv_match(
                    type=["GLOBAL_POSITION_INT", "ATTITUDE", "NAV_CONTROLLER_OUTPUT"],         
                    blocking=False
                    )
                    if m is None:
                        break
                    t = m.get_type()
                    if t == "GLOBAL_POSITION_INT":
                        last_gp = m
                    elif t == "ATTITUDE":
                        last_att = m
                    elif t == "NAV_CONTROLLER_OUTPUT":
                        last_nav = m
                         
                if last_gp is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                if now - last_step < cmd_dt:
                    time.sleep(0.001)
                    continue
                last_step = now

                rel_alt = last_gp.relative_alt / 1000.0

                # wind enable
                if (not wind_enabled) and (rel_alt >= args.wind_enable_alt) and (args.wind_speed_mps > 0):
                    wind_enabled = True
                    wind_enable_time = now
                    set_param(master, "SIM_WIND_TURB", args.wind_turb)
                    set_param(master, "SIM_WIND_DIR", args.wind_from_deg)
                    wind_speed_now = 0.0

                if wind_enabled:
                    if args.wind_ramp_s <= 0.0:
                        wind_speed_now = args.wind_speed_mps
                    else:
                        elapsed = max(0.0, now - wind_enable_time)
                        wind_speed_now = clamp(args.wind_speed_mps * (elapsed / args.wind_ramp_s), 0.0, args.wind_speed_mps)
                    set_param(master, "SIM_WIND_SPD", wind_speed_now)

                # start spiral
                if (not spiral_started) and (rel_alt >= args.spiral_start_alt):
                    spiral_started = True
                    h_start = rel_alt

                    lat = last_gp.lat / 1e7
                    lon = last_gp.lon / 1e7

                    radial = math.radians(args.start_radial_deg)
                    north_vec = args.r0 * math.cos(radial)
                    east_vec  = args.r0 * math.sin(radial)
                    center_lat, center_lon = shift_latlon(lat, lon, -north_vec, -east_vec)

                    n_now, e_now = latlon_to_ne(lat, lon, center_lat, center_lon)
                    phi_prev = math.atan2(e_now, n_now)
                    phi_accum = 0.0
                    turns_completed_monotone = 0.0

                    try:
                        master.set_mode("GUIDED")
                    except Exception:
                        pass
                    send_guided_change_speed(master, args.speed)

                    print(f"[SPIRAL] start @ {h_start:.1f}m  algo={args.algo}")

                if not spiral_started:
                    continue

                # actual position relative to center
                lat = last_gp.lat / 1e7
                lon = last_gp.lon / 1e7
                n_now, e_now = latlon_to_ne(lat, lon, center_lat, center_lon)
                r_now = math.hypot(n_now, e_now)
                phi = math.atan2(e_now, n_now)

                dphi = wrap_pi(phi - phi_prev)
                phi_prev = phi
                phi_accum += dphi

                turns_completed = abs(phi_accum) / (2.0 * math.pi)
                if turns_completed < turns_completed_monotone:
                    turns_completed = turns_completed_monotone
                else:
                    turns_completed_monotone = turns_completed

                if turns_completed >= total_turns:
                    turns_completed = total_turns
                    done = True

                r_ref = args.r0 + args.r_step * turns_completed
                progress = turns_completed / max(1e-6, total_turns)
                h_ref = h_start + args.spiral_climb * progress

                look = math.radians(args.lookahead_deg)
                phi_ref = phi  # 不加lookahead
                ref_n = r_ref * math.cos(phi_ref)
                ref_e = r_ref * math.sin(phi_ref)
                ref_lat, ref_lon = ne_to_latlon(ref_n, ref_e, center_lat, center_lon)
                ref_heading = wrap_360(math.degrees(phi_ref + dir_sign * (math.pi/2.0)))
                
                phi_tgt = phi + dir_sign * look

                tgt_n = r_ref * math.cos(phi_tgt)
                tgt_e = r_ref * math.sin(phi_tgt)
                tgt_lat, tgt_lon = ne_to_latlon(tgt_n, tgt_e, center_lat, center_lon)

                phi_heading = phi_tgt + dir_sign * (math.pi / 2.0)
                tgt_heading = wrap_360(math.degrees(phi_heading))

                # radial error
                dr = r_now - r_ref
                absd = abs(dr)

                s = state_from_dr(dr, args.deadband)
                a = "U"  # baseline default
                if args.algo == "ql":
                    a = greedy_action(Q, s)
                    if a == "D":
                        K += args.K_step
                    elif a == "I":
                        K -= args.K_step
                    K = clamp(K, args.K_min, args.K_max)

                # control correction
                n_err, e_err = ne_error_from_latlon(lat, lon, tgt_lat, tgt_lon)
                e_ct = cross_track_error(n_err, e_err, tgt_heading)

                delta_deg = math.degrees(math.atan2(e_ct, max(1e-3, args.L_corr))) * K
                delta_deg = clamp(delta_deg, -args.max_corr_deg, args.max_corr_deg)
                cmd_heading = wrap_360(tgt_heading + delta_deg)

                send_guided_goto(master, tgt_lat, tgt_lon, h_ref)
                send_guided_change_heading(master, cmd_heading, max_centrip_acc=args.centrip_acc)
                send_guided_change_alt(master, h_ref, rate_mps=args.climb_rate)

                rwd = reward_from_error(absd, args.e_expected)

                abs_dr.append(absd)
                dr_list.append(dr)
                n_metric += 1
                max_abs = max(max_abs, absd)
                if absd <= 5.0:
                    within5 += 1
                    
                # --- attitude in degrees (each row) ---
                if last_att is not None:
                    roll_deg  = math.degrees(last_att.roll)
                    pitch_deg = math.degrees(last_att.pitch)
                    yaw_deg   = wrap_360(math.degrees(last_att.yaw))
                else:
                    roll_deg = pitch_deg = yaw_deg = ""

                w.writerow([
                    now - t0, args.algo, K, s, a,
                    dr, absd, rwd,
                    lat, lon, rel_alt,
                    ref_lat, ref_lon, h_ref, ref_heading,
                    tgt_lat, tgt_lon, h_ref,
                    tgt_heading, cmd_heading,
                    r_ref, r_now, turns_completed, progress,
                    wind_speed_now if wind_enabled else 0.0, args.wind_from_deg,
                    roll_deg, pitch_deg, yaw_deg
                ])
                f.flush()

        except KeyboardInterrupt:
            print("\nStopped by user.")

    summary = {
        "algo": args.algo,
        "qtable": os.path.basename(args.qtable) if args.qtable else None,
        "wind_speed_mps": args.wind_speed_mps,
        "wind_from_deg": args.wind_from_deg,
        "r0": args.r0,
        "r_step": args.r_step,
        "turns": args.turns,
        "end_radius": args.r0 + args.r_step * args.turns,
        "mean_abs_dr": mean(abs_dr) if abs_dr else None,
        "rms_dr": rms(dr_list) if dr_list else None,
        "max_abs_dr": max_abs if n_metric > 0 else None,
        "pct_within_5m": (within5 / n_metric) if n_metric > 0 else None,
        "count": n_metric
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== TEST SUMMARY ===")
    print(summary)
    print("Saved log:", args.out_csv)
    print("Saved summary:", args.out_summary)

if __name__ == "__main__":
    main()

