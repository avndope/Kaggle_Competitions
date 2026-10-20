"""Read v13's own Kaggle agent logs: identification, exploit gain and timing on real hardware.

    python3 ladder_logs.py SUBMISSION_ID [N_SAMPLE]

Kaggle serves each agent's per-step log (duration, stdout, stderr) at
/api/v1/competitions/episodes/{episodeId}/agents/{agentIndex}/logs. v13 prints one
line at step 716: `V13 <bundle> alive=[...] exploited=N gain=G max_turn=T`.
Read-only: nothing is written to disk.
"""
import collections
import json
import re
import sys
import time
import urllib.request

TOK = open("/mnt/c/Users/avnbe/.kaggle/access_token").read().strip()
H = {"Authorization": "Bearer " + TOK, "Content-Type": "application/json"}


def post(svc, body):
    req = urllib.request.Request("https://www.kaggle.com/api/i/" + svc, data=json.dumps(body).encode(),
                                 method="POST", headers=H)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def agent_logs(eid, idx):
    url = "https://www.kaggle.com/api/v1/competitions/episodes/%d/agents/%d/logs" % (eid, idx)
    for a in range(5):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=120).read())
        except Exception:
            time.sleep(4 * 2 ** a)
    return None


def main():
    sub = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    eps = sorted(post("competitions.EpisodeService/ListEpisodes", {"submissionId": sub})["episodes"],
                 key=lambda e: e["createTime"])
    games = [e for e in eps if len({a["submissionId"] for a in e["agents"]}) == 2]
    sample = games[::max(1, len(games) // n)][-n:]
    alive, rows = collections.Counter(), []
    for e in sample:
        me = [a for a in e["agents"] if a["submissionId"] == sub][0]
        op = [a for a in e["agents"] if a["submissionId"] != sub][0]
        logs = agent_logs(e["id"], me.get("index", 0))
        if not logs:
            continue
        d = [(s[0].get("duration") or 0) if s and isinstance(s, list) else 0 for s in logs]
        line = [s[0].get("stdout") for s in logs if s and s[0].get("stdout") and "V13" in s[0].get("stdout")]
        m = re.search(r"alive=(\[.*?\]) exploited=(\d+) gain=(-?\d+)", line[0]) if line else None
        al = json.loads(m.group(1).replace("'", '"')) if m else None
        for a in (al or ["(none)"]):
            alive[a] += 1
        rows.append((e["createTime"][5:16], d[0], max(d), sum(x > 1 for x in d), sum(max(0, x - 1) for x in d),
                     (me.get("reward") or 0) - (op.get("reward") or 0), op.get("initialScore") or 0,
                     al, int(m.group(3)) if m else None))
    print("%-12s %5s %5s %4s %5s %8s %6s  %s" % ("time", "t0", "max", ">1s", "over", "margin", "oppR", "alive / gain"))
    for r in rows:
        print("%-12s %5.2f %5.2f %4d %5.2f %+8.0f %6.0f  %s gain=%s" % r)
    ident = [r for r in rows if r[7]]
    print("\nidentified %d/%d; wins when identified %d/%d; alive counts %s"
          % (len(ident), len(rows), sum(r[5] > 0 for r in ident), len(ident), dict(alive)))
    print("turn-0 duration mean %.2fs max %.2fs; overage per game mean %.2fs max %.2fs (bank 60 s)"
          % (sum(r[1] for r in rows) / len(rows), max(r[1] for r in rows),
             sum(r[4] for r in rows) / len(rows), max(r[4] for r in rows)))


if __name__ == "__main__":
    main()
