"""Scan Kaggriculture for new or updated public notebooks and discussion activity.

    python3 scan_new.py [--since 2026-09-30T20:00]

Compares against scan_state.json (same folder): prints notebooks whose lastRunTime
is newer than the last scan (or --since) and discussion topics that are new or have
new comments, then updates the state. Read-only against Kaggle; nothing downloaded
is executed.
"""
import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "scan_state.json")
TOK = open("/mnt/c/Users/avnbe/.kaggle/access_token").read().strip()
H = {"Authorization": "Bearer " + TOK, "Content-Type": "application/json"}
FORUM = 11548702


def post(svc, body):
    req = urllib.request.Request("https://www.kaggle.com/api/i/" + svc, data=json.dumps(body).encode(),
                                 method="POST", headers=H)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    args = ap.parse_args()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    since = args.since or state.get("last_scan", "2026-09-30T20:00")

    kernels = []
    for page in (1, 2):
        kernels += get("https://www.kaggle.com/api/v1/kernels/list?competition=kaggriculture"
                       "&sortBy=dateRun&pageSize=50&page=%d" % page)
    new_k = [k for k in kernels if (k.get("lastRunTime") or "") > since]
    seen_k = set(state.get("kernels", []))

    topics = post("discussions.DiscussionsService/GetTopicListByForumId", {"forumId": FORUM, "page": 1}).get("topics", [])
    new_t = [t for t in topics if max(t.get("postDate") or "", t.get("lastCommentPostDate") or "") > since]

    print("scan since %s UTC" % since[:16])
    print("notebooks run/updated: %d" % len(new_k))
    for k in sorted(new_k, key=lambda k: k["lastRunTime"], reverse=True):
        tag = "UPDATED" if k["ref"] in seen_k else "NEW"
        print("  %-7s %s  %-70s votes %s  %s" % (tag, k["lastRunTime"][11:16], k["ref"][:70], k.get("totalVotes"),
                                              (k.get("title") or "")[:60]))
    print("discussion topics new/active: %d" % len(new_t))
    for t in sorted(new_t, key=lambda t: max(t.get("postDate") or "", t.get("lastCommentPostDate") or ""), reverse=True):
        new_topic = (t.get("postDate") or "") > since
        print("  %-7s %s  comments %-3s %s  (last: %s)  https://www.kaggle.com/competitions/kaggriculture/discussion/%s"
              % ("NEW" if new_topic else "COMMENT", max(t.get("postDate") or "", t.get("lastCommentPostDate") or "")[11:16],
                 t.get("commentCount"), (t.get("title") or "")[:70], t.get("lastCommenterName"), t.get("id")))

    stamp = max([since] + [k["lastRunTime"] for k in kernels if k.get("lastRunTime")] +
                [max(t.get("postDate") or "", t.get("lastCommentPostDate") or "") for t in topics])
    state["last_scan"] = stamp
    state["kernels"] = sorted(seen_k | {k["ref"] for k in kernels})
    json.dump(state, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
