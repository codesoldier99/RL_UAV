#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced spiral comparison test with multiple baselines and altitude tracking.

This script extends test_spiral_compare_fixed.py with:
  - PID adaptive gain controller (baseline_pid)
  - Rule-based piecewise gain controller (baseline_rule)
  - Altitude tracking error metrics (MAE, RMSE for height)
  - Gain K variation logging for convergence analysis
  - Support for state-space ablation (3/5/7/9 discrete states)

Usage examples:
  # Q-learning controller
  python test_spiral_compare_enhanced.py --algo ql --qtable qtable_final_5m_fixed.json

  # Original fixed-gain baseline
  python test_spiral_compare_enhanced.py --algo baseline

  # PID adaptive gain baseline
  python test_spiral_compare_enhanced.py --algo baseline_pid

  # Rule-based piecewise gain baseline
  python test_spiral_compare_enhanced.py --algo baseline_rule

  # 5-state ablation Q-learning (requires corresponding Q-table)
  python test_spiral_compare_enhanced.py --algo ql --qtable qtable_5state.json --n_states 5
"""

import argparse, csv, json, math, os, time
from pymavlink import mavutil

M_PER_DEG = 111319.49079327357

# Default 3-state definition (can be overridden by --n_states)
STATES_3  = ["L", "M", "R"]
ACTIONS = ["D", "U", "I"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_360(deg):
    deg = deg % 360.0
    return deg if deg >= 0 else deg + 360.0


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def latlon_to_ne(lat_deg, lon_deg, lat0_deg, lon0_deg):
    dlat = lat_deg - lat0_deg
    dlon = lon_deg - lon0_deg
    north = dlat * M_PER_DEG
    east = dlon * M_PER_DEG * math.cos(math.radians(lat0_deg))
    return north, east


def ne_to_latlon(north_m, east_m, lat0_deg, lon0_deg):
    dlat = north_m / M_PER_DEG
    dlon = east_m / (M_PER_DEG * math.cos(math.radians(lat0_deg)))
    return lat0_deg + dlat, lon0_deg + dlon


def shift_latlon(lat_deg, lon_deg, north_m, east_m):
    return ne_to_latlon(north_m, east_m, lat_deg, lon_deg)


def ne_error_from_latlon(lat_deg, lon_deg, tgt_lat_deg, tgt_lon_deg):
    lat_rad = math.radians(lat_deg)
    dlat = tgt_lat_deg - lat_deg
    dlon = tgt_lon_deg - lon_deg
    north = dlat * M_PER_DEG
    east = dlon * M_PER_DEG * math.cos(lat_rad)
    return north, east


def cross_track_error(n_err, e_err, tgt_heading_deg):
    chi = math.radians(wrap_360(tgt_heading_deg))
    right_n = -math.sin(chi)
    right_e = math.cos(chi)
    return n_err * right_n + e_err * right_e


# ---------------------------------------------------------------------------
# State discretisation helpers (supports 3, 5, 7, 9 states for ablation)
# ---------------------------------------------------------------------------

def build_state_labels(n_states):
    """Return ordered state labels for *n_states* discrete states.

    For n_states=3 → ["L", "M", "R"]  (original)
    For n_states=5 → ["LL", "L", "M", "R", "RR"]
    For n_states=7 → ["LLL", "LL", "L", "M", "R", "RR", "RRR"]
    For n_states=9 → ["LLLL", "LLL", "LL", "L", "M", "R", "RR", "RRR", "RRRR"]
    """
    if n_states == 3:
        return ["L", "M", "R"]
    half = (n_states - 1) // 2
    labels = []
    for i in range(half, 0, -1):
        labels.append("L" * i)
    labels.append("M")
    for i in range(1, half + 1):
        labels.append("R" * i)
    return labels


def build_state_thresholds(n_states, deadband):
    """Return ascending threshold list that divides the radial error axis.

    For n_states=3, deadband=5:
        thresholds = [-5, 5]  → <-5 ⇒ L, [-5,5] ⇒ M, >5 ⇒ R

    For n_states=5, deadband=5:
        thresholds = [-10, -5, 5, 10]
    """
    half = (n_states - 1) // 2
    # inner thresholds centred on zero
    thresholds = []
    for i in range(half, 0, -1):
        thresholds.append(-deadband * i)
    for i in range(1, half + 1):
        thresholds.append(deadband * i)
    return thresholds


def state_from_dr_n(dr_m, thresholds, labels):
    """Map continuous radial error to one of *n_states* discrete states."""
    for i, thr in enumerate(thresholds):
        if dr_m < thr:
            return labels[i]
    return labels[-1]


# Original 3-state helper retained for backward compatibility
def state_from_dr(dr_m, deadband):
    if dr_m < -deadband:
        return "L"
    if dr_m > deadband:
        return "R"
    return "M"


def reward_from_error(err_abs, e_expected, r_pos=100, r_neg=-1):
    """Sparse reward: +r_pos if |error| ≤ e_expected, else r_neg."""
    return r_pos if err_abs <= e_expected else r_neg


def greedy_action(Q, s):
    row = Q.get(s, {})
    vals = {a: float(row.get(a, 0.0)) for a in ACTIONS}
    best_val = max(vals.values())
    best = [a for a, v in vals.items() if v == best_val]
    if "U" in best:
        return "U"
    return best[0]


def load_qtable(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    Q = obj.get("Q", None)
    if Q is None:
        raise ValueError("Q-table JSON has no 'Q' field.")
    return Q


# ---------------------------------------------------------------------------
# PID adaptive gain controller
# ---------------------------------------------------------------------------

class PIDAdaptiveGain:
    """A PID controller that adjusts the navigation gain K based on
    the radial tracking error.  This serves as a stronger baseline
    compared to a simple fixed-gain controller.

    K_cmd = K_base + Kp*|e| + Ki*∫|e|dt + Kd*(d|e|/dt)
    Output K is clamped to [K_min, K_max].
    """

    def __init__(self, K_base=1.0, Kp=0.08, Ki=0.005, Kd=0.02,
                 K_min=0.5, K_max=2.0, dt=0.2, integral_limit=50.0):
        self.K_base = K_base
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.K_min = K_min
        self.K_max = K_max
        self.dt = dt
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_err = 0.0

    def update(self, abs_error):
        self._integral += abs_error * self.dt
        self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        derivative = (abs_error - self._prev_err) / max(self.dt, 1e-6)
        self._prev_err = abs_error
        K = self.K_base + self.Kp * abs_error + self.Ki * self._integral + self.Kd * derivative
        return clamp(K, self.K_min, self.K_max)


# ---------------------------------------------------------------------------
# Rule-based piecewise gain controller
# ---------------------------------------------------------------------------

def rule_based_gain(abs_error, K_min=0.5, K_max=2.0):
    """Simple piecewise-linear gain schedule based on error magnitude.

    |error| < 2 m   → K = K_min  (small correction)
    2 ≤ |error| < 5 → K linearly increases from K_min to 1.2
    5 ≤ |error| < 10 → K linearly increases from 1.2 to 1.8
    |error| ≥ 10    → K = K_max  (maximum correction)
    """
    if abs_error < 2.0:
        return K_min
    elif abs_error < 5.0:
        return K_min + (1.2 - K_min) * (abs_error - 2.0) / 3.0
    elif abs_error < 10.0:
        return 1.2 + (1.8 - 1.2) * (abs_error - 5.0) / 5.0
    else:
        return K_max


# ---------------------------------------------------------------------------
# MAVLink helpers (identical to original)
# ---------------------------------------------------------------------------

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
        float(alt_rel_m),
    )


def send_guided_change_speed(master, groundspeed_mps):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43000, 0, 0,
        1, float(groundspeed_mps), 0.0, 0,
        0, 0, 0,
    )


def send_guided_change_heading(master, heading_deg, max_centrip_acc=2.0):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43002, 0, 0,
        0, float(heading_deg), float(max_centrip_acc), 0,
        0, 0, 0,
    )


def send_guided_change_alt(master, target_alt_m, rate_mps=2.0):
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43001, 0, 0,
        0, 0, float(rate_mps), 0,
        0, 0, float(target_alt_m),
    )


def set_param(master, name: str, value: float):
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rms(xs):
    return math.sqrt(sum([x * x for x in xs]) / max(1, len(xs)))


def main():
    ap = argparse.ArgumentParser(
        description="Enhanced spiral tracking comparison with multiple baselines."
    )

    ap.add_argument("--master", default="udp:127.0.0.1:14550")

    ap.add_argument(
        "--algo",
        choices=["baseline", "baseline_pid", "baseline_rule", "ql"],
        default="ql",
        help=(
            "baseline       = fixed-gain K (original);\n"
            "baseline_pid   = PID adaptive gain;\n"
            "baseline_rule  = rule-based piecewise gain;\n"
            "ql             = Q-learning table lookup"
        ),
    )

    ap.add_argument("--qtable", default="", help="Q-table JSON path (required for algo=ql)")
    ap.add_argument("--out_csv", default="test_log.csv")
    ap.add_argument("--out_summary", default="test_summary.json")

    # Spiral parameters
    ap.add_argument("--turns", type=float, default=3.0)
    ap.add_argument("--r0", type=float, default=80.0)
    ap.add_argument("--r_step", type=float, default=80.0)
    ap.add_argument("--direction", choices=["CW", "CCW"], default="CW")

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

    # Eval thresholds
    ap.add_argument("--deadband", type=float, default=5.0,
                    help="Threshold epsilon for state discretisation (metres)")
    ap.add_argument("--e_expected", type=float, default=5.0)

    # Gain K
    ap.add_argument("--K_fixed", type=float, default=0.8, help="Fixed K for baseline")
    ap.add_argument("--K_init", type=float, default=1.0, help="Initial K for ql/pid")
    ap.add_argument("--K_step", type=float, default=0.10,
                    help="Gain adjustment step per Q-learning action (ΔK)")
    ap.add_argument("--K_min", type=float, default=0.5)
    ap.add_argument("--K_max", type=float, default=2.0)

    # Control
    ap.add_argument("--L_corr", type=float, default=50.0)
    ap.add_argument("--max_corr_deg", type=float, default=25.0)
    ap.add_argument("--lookahead_deg", type=float, default=15.0)

    # Center offset
    ap.add_argument("--start_radial_deg", type=float, default=90.0)

    # State-space ablation
    ap.add_argument("--n_states", type=int, default=3, choices=[3, 5, 7, 9],
                    help="Number of discrete states for ablation study")

    args = ap.parse_args()

    # Build state labels and thresholds for the chosen granularity
    state_labels = build_state_labels(args.n_states)
    state_thresholds = build_state_thresholds(args.n_states, args.deadband)

    # Controller setup
    Q = None
    pid_ctrl = None

    if args.algo == "ql":
        if not args.qtable:
            raise SystemExit("algo=ql requires --qtable path")
        Q = load_qtable(args.qtable)
        # Ensure all state-action pairs present
        for s in state_labels:
            Q.setdefault(s, {})
            for a in ACTIONS:
                Q[s].setdefault(a, 0.0)
        print(f"[QTABLE] loaded: {args.qtable}  states={state_labels}")
        print(f"[QTABLE] policy: { {s: greedy_action(Q, s) for s in state_labels} }")
    elif args.algo == "baseline_pid":
        pid_ctrl = PIDAdaptiveGain(
            K_base=args.K_init,
            K_min=args.K_min,
            K_max=args.K_max,
            dt=1.0 / args.cmd_hz,
        )
        print(f"[BASELINE_PID] PID adaptive gain  K_base={args.K_init}")
    elif args.algo == "baseline_rule":
        print(f"[BASELINE_RULE] rule-based piecewise gain  K_min={args.K_min} K_max={args.K_max}")
    else:
        print(f"[BASELINE] fixed K = {args.K_fixed}")

    master = mavutil.mavlink_connection(args.master)
    master.wait_heartbeat()
    print("Heartbeat OK")

    # Reset wind
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
    last_att = None
    last_nav = None

    # Radial error accumulators
    abs_dr, dr_list = [], []
    within5, n_metric, max_abs = 0, 0, 0.0

    # Altitude error accumulators  (P0 revision: altitude tracking metrics)
    alt_errors = []  # signed altitude error (actual - reference)

    # Gain history for convergence visualisation (P1 revision)
    gain_history = []  # list of (time_s, K)

    t0 = time.time()
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s", "algo", "K", "state", "action",
            "dr_m", "abs_dr_m", "reward",
            "lat_deg", "lon_deg", "rel_alt_m",
            "ref_lat_deg", "ref_lon_deg", "ref_alt_m", "ref_heading_deg",
            "tgt_lat_deg", "tgt_lon_deg", "tgt_alt_m",
            "tgt_heading_deg", "cmd_heading_deg",
            "r_ref_m", "r_now_m", "turns_completed", "progress",
            "wind_speed_mps", "wind_from_deg",
            "roll_deg", "pitch_deg", "yaw_deg",
            "alt_error_m",  # NEW: altitude tracking error
        ])

        print("Running... Ctrl+C to stop.")
        done = False
        try:
            while not done:
                for _ in range(20):
                    m = master.recv_match(
                        type=["GLOBAL_POSITION_INT", "ATTITUDE", "NAV_CONTROLLER_OUTPUT"],
                        blocking=False,
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

                # Wind enable
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
                        wind_speed_now = clamp(
                            args.wind_speed_mps * (elapsed / args.wind_ramp_s),
                            0.0, args.wind_speed_mps,
                        )
                    set_param(master, "SIM_WIND_SPD", wind_speed_now)

                # Start spiral
                if (not spiral_started) and (rel_alt >= args.spiral_start_alt):
                    spiral_started = True
                    h_start = rel_alt

                    lat = last_gp.lat / 1e7
                    lon = last_gp.lon / 1e7

                    radial = math.radians(args.start_radial_deg)
                    north_vec = args.r0 * math.cos(radial)
                    east_vec = args.r0 * math.sin(radial)
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

                # Position relative to center
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
                phi_ref = phi
                ref_n = r_ref * math.cos(phi_ref)
                ref_e = r_ref * math.sin(phi_ref)
                ref_lat, ref_lon = ne_to_latlon(ref_n, ref_e, center_lat, center_lon)
                ref_heading = wrap_360(math.degrees(phi_ref + dir_sign * (math.pi / 2.0)))

                phi_tgt = phi + dir_sign * look
                tgt_n = r_ref * math.cos(phi_tgt)
                tgt_e = r_ref * math.sin(phi_tgt)
                tgt_lat, tgt_lon = ne_to_latlon(tgt_n, tgt_e, center_lat, center_lon)

                phi_heading = phi_tgt + dir_sign * (math.pi / 2.0)
                tgt_heading = wrap_360(math.degrees(phi_heading))

                # Radial error
                dr = r_now - r_ref
                absd = abs(dr)

                # Altitude error (P0: altitude tracking)
                alt_err = rel_alt - h_ref

                # ----- Controller action -----
                a = "U"  # default
                if args.algo == "ql":
                    if args.n_states == 3:
                        s = state_from_dr(dr, args.deadband)
                    else:
                        s = state_from_dr_n(dr, state_thresholds, state_labels)
                    a = greedy_action(Q, s)
                    if a == "D":
                        K += args.K_step
                    elif a == "I":
                        K -= args.K_step
                    K = clamp(K, args.K_min, args.K_max)

                elif args.algo == "baseline_pid":
                    s = state_from_dr(dr, args.deadband)
                    K = pid_ctrl.update(absd)

                elif args.algo == "baseline_rule":
                    s = state_from_dr(dr, args.deadband)
                    K = rule_based_gain(absd, K_min=args.K_min, K_max=args.K_max)

                else:  # baseline fixed
                    s = state_from_dr(dr, args.deadband)
                    K = args.K_fixed

                # Control correction
                n_err, e_err = ne_error_from_latlon(lat, lon, tgt_lat, tgt_lon)
                e_ct = cross_track_error(n_err, e_err, tgt_heading)

                delta_deg = math.degrees(math.atan2(e_ct, max(1e-3, args.L_corr))) * K
                delta_deg = clamp(delta_deg, -args.max_corr_deg, args.max_corr_deg)
                cmd_heading = wrap_360(tgt_heading + delta_deg)

                send_guided_goto(master, tgt_lat, tgt_lon, h_ref)
                send_guided_change_heading(master, cmd_heading, max_centrip_acc=args.centrip_acc)
                send_guided_change_alt(master, h_ref, rate_mps=args.climb_rate)

                rwd = reward_from_error(absd, args.e_expected)

                # Accumulate metrics
                elapsed_s = now - t0
                abs_dr.append(absd)
                dr_list.append(dr)
                n_metric += 1
                max_abs = max(max_abs, absd)
                if absd <= 5.0:
                    within5 += 1

                alt_errors.append(alt_err)
                gain_history.append((elapsed_s, K))

                # Attitude
                if last_att is not None:
                    roll_deg = math.degrees(last_att.roll)
                    pitch_deg = math.degrees(last_att.pitch)
                    yaw_deg = wrap_360(math.degrees(last_att.yaw))
                else:
                    roll_deg = pitch_deg = yaw_deg = ""

                w.writerow([
                    elapsed_s, args.algo, K, s, a,
                    dr, absd, rwd,
                    lat, lon, rel_alt,
                    ref_lat, ref_lon, h_ref, ref_heading,
                    tgt_lat, tgt_lon, h_ref,
                    tgt_heading, cmd_heading,
                    r_ref, r_now, turns_completed, progress,
                    wind_speed_now if wind_enabled else 0.0, args.wind_from_deg,
                    roll_deg, pitch_deg, yaw_deg,
                    alt_err,
                ])
                f.flush()

        except KeyboardInterrupt:
            print("\nStopped by user.")

    # ----- Summary with altitude metrics -----
    abs_alt_errors = [abs(e) for e in alt_errors]
    summary = {
        "algo": args.algo,
        "n_states": args.n_states,
        "qtable": os.path.basename(args.qtable) if args.qtable else None,
        "wind_speed_mps": args.wind_speed_mps,
        "wind_from_deg": args.wind_from_deg,
        "r0": args.r0,
        "r_step": args.r_step,
        "turns": args.turns,
        "end_radius": args.r0 + args.r_step * args.turns,
        # Radial tracking metrics
        "radial_MAE": mean(abs_dr) if abs_dr else None,
        "radial_RMSE": rms(dr_list) if dr_list else None,
        "radial_max_abs": max_abs if n_metric > 0 else None,
        "pct_within_5m": (within5 / n_metric) if n_metric > 0 else None,
        # Altitude tracking metrics (P0 revision)
        "altitude_MAE": mean(abs_alt_errors) if abs_alt_errors else None,
        "altitude_RMSE": rms(alt_errors) if alt_errors else None,
        "altitude_max_abs": max(abs_alt_errors) if abs_alt_errors else None,
        "count": n_metric,
        # Gain history summary
        "K_final": K,
        "K_mean": mean([g[1] for g in gain_history]) if gain_history else None,
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save gain history for plotting
    gain_csv = args.out_csv.replace(".csv", "_gain_history.csv")
    with open(gain_csv, "w", newline="", encoding="utf-8") as f:
        gw = csv.writer(f)
        gw.writerow(["time_s", "K"])
        for t_s, k_val in gain_history:
            gw.writerow([t_s, k_val])

    print("\n=== TEST SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved log:          {args.out_csv}")
    print(f"Saved summary:      {args.out_summary}")
    print(f"Saved gain history: {gain_csv}")


if __name__ == "__main__":
    main()
