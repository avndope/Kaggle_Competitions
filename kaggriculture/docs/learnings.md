# Kaggriculture — learnings

Kaggle simulation competition, Jul–Oct 2026. Two players run farms for 720 turns (30 days × 24 hours), and whoever ends with more money wins. Submissions closed on 30 Sep 2026; the final ranking is one Bradley-Terry tournament over the games played until 14 Oct.

**Our result:** rank **1,432 of 10,148** (top 14.1%), no medal, on the private leaderboard published after 14 Oct.
- Our counted score was v16's 1732.1; v14 (re-uploaded) ended at 1716.2.
- After the deadline v16 went 393–387 and v14 394–389.
- Cutoffs: bronze 1843.4 (rank 1,014), silver 2093.1 (rank 507), gold 2768.3 (rank 30).
- We peaked at rank 513 on 26 Sep, slid to 1,614 by 8 Oct as the field moved to learned policies, then recovered.

This file records what the leaders built (architecture, methods, algorithms), what we learned first-hand, and a playbook for the next simulation competition. Leader ranks are as stated in their own posts, before the final tournament.

---

## 1. Game mechanics that shaped the methods

These mechanics decided which methods worked, so they come first.

- **Shared, deep-or-thin market.**
  - Prices follow each product's market inventory. Egg is deep; strawberry, melon and tomato are thin.
  - Both players trade against the same inventory, so every sale lowers the price for the opponent too.
- **Per-unit lockstep order fills.**
  - A market order fills one unit at a time.
  - Both players' i-th units get the same quote from the same pre-commit inventory.
  - An order stops at its first failing unit (empty shed, or not enough money), and the rest is silently dropped.
  - Earlier order slots therefore get better quotes.
- **No free round trips.** Buys are quoted at the post-buy inventory "so a buy/sell round-trip against an unchanged market nets zero" (engine comment). There is no arbitrage against an unchanged market.
- **Order slots.** At most 10 market orders per turn, and hires use the same slots.
- **Hires.** The n-th hire of a day costs fib(n), and hands vanish at the end of the day.
- **The shed holds 100 items.**
  - At night, everything units carry is dumped into the shed, and whatever doesn't fit is destroyed.
  - On day 29 there is no night unload, so goods still carried at the end are lost.
- **Land.** Quadrants cost 1,000, 2,000 and 4,000. The top players owned all four by about day 10.
- **Board RNG.** Weed spawns and shop unlocks share one RNG per day, and the draws depend on each farm's empty tiles. Any change in behaviour reshuffles the rest of the game.
- **The engine is deterministic given the seed.** That makes paired-seed testing and exact replay reproduction possible.
- **1 s of CPU per turn**, with a small loading allowance on turn 0. Kaggle loads the **last callable** in the agent file.
- **Two submission slots.** Only the latest two count (`numScoredSubmissions: 2`), and each new upload retires the older active one.

---

## 2. Methods by rank

| Rank (own post) | Team | Method in one line |
|---|---|---|
| 1st | M & M & P & Q | Behaviour cloning, then self-play PPO in repeated cycles (8.29M games); a shared Transformer for units and market; rule-based patches at inference |
| 4th | Yaroslav Pudovkin | Pure behaviour cloning, no RL or search; a one-pass transformer drafts the whole turn and a corrector revises it; trained on executed state changes |
| 7th | DECEM | Behaviour cloning only: an 8.2M entity Transformer conditioned on team, version and outcome; units decoded in turn; two small rules |
| 10th | MarvinTMB | Behaviour cloning, then PPO in a self-play league, with a small (~947k) actor and a LeJEPA backbone; 10 h on one RTX 5090 |
| top 10 | Russell Kirk | Supervised fine-tuning on winners' actions, generated token by token like text |
| 31st (24th on 4 Oct) | t fuku / PavelSavchenkov | A cloned network outputs one daily intent; a C++ compiler turns it into the day's actions; plus a learned forecast of the opponent's sales |
| ~55th | Snorlax | Cloning warm start, then PPO on daily per-tile plans; a Rust planner executes them |
| 83rd | Syed Asad Ali | Exact simulator, rule-based baseline, imitation learning, then RL fine-tuning |
| ~777th | Takamichi Toda | Rule-based agent with "fly brain" alarm circuits |
| **~1,600th** | **us** | Strongest public heuristic agent, plus in-game opponent identification and same-turn market exploitation |

**The pattern:**
- Every top-10 entry learned a turn-level policy from recorded top games (behaviour cloning, called BC from here on).
- RL on top added a little, and only with very large compute.
- Public heuristic agents and their forks, including ours, ended in the 1,600–1,900 rating band, which became a crowd of near-identical clones.
- A strong private heuristic planner could still compete: the 1st-place team's standalone planner reached #1 at one point.

---

## 3. Leader solutions in detail

### 3.1 1st place: BC, self-play PPO and heuristics (M & M & P & Q)
Source: [discussion 745073](https://www.kaggle.com/competitions/kaggriculture/discussion/745073), code at [github.com/msdsm/kaggriculture-solution](https://github.com/msdsm/kaggriculture-solution).

- **Training loop.** They alternated BC and self-play PPO.
  - BC data came from public replays plus demonstrations generated by their own heuristic planner.
  - They studied replays to find weaknesses, improved the planner, and generated targeted demonstrations for the next cycle.
- **Scale.**
  - The final 10M-parameter model's lineage includes about **8.29M self-play games**.
  - Compute: up to 17 A100 + 29 A30 for the 10M PPO run (about 7 A100 + 20 A30 on average), and up to 26 A100 for a parallel 20M line.
  - The team tested two hypotheses side by side: more self-play games vs more parameters.
- **Architecture.** A shared Transformer predicts both unit actions and market orders.
- **Inference.**
  - "Final A" adds a rule-based action patch at inference that resolves conflicts, avoids wasteful actions and manages inventory.
  - "Final B" is trained with rule-aware PPO and sequential action masks.
  - **B was clearly better.** Against the 2nd-place team's two submissions, B scored 57% and 93%; A scored 37% and 76%.
  - Lesson: build the rules into training, not just into inference.
- **BC after PPO.** Re-running BC on a PPO-trained model first lost 100 of 100 games to the pre-BC checkpoint. Resuming PPO from it then overtook the old checkpoint after about 60k games. A temporary drop after re-cloning is normal.
- **A strong heuristic planner mattered in two ways.** It reached #1 on its own at one point, and it served as a demonstration generator.

### 3.2 4th place: no RL, no search; a one-pass transformer
Source: [discussion 745119](https://www.kaggle.com/competitions/kaggriculture/discussion/745119). The most detailed write-up; worth re-reading in full.

- **Imitation target: what the engine executed, not what the player requested.**
  - Top players send oversized orders and let the engine clip them. One player had only 30% of its requested sells filled; two others about 2%.
  - So they replayed every game through the engine with a bit-exact money check per step, kept only executed trades, and netted out same-turn round trips.
  - All input features were computed from the same replayed record as the labels.
  - Effect: a clone that won 3/24 against their previous best won 12/24; adding a feed guard made it 23/24.
- **Staged training.**
  1. Pretrain on many top teams' relabelled games.
  2. Fine-tune on one strong team: +$9.2k, 89/96 wins vs the previous model, and better than training on that team alone from scratch.
  3. Average the weights of two fine-tunes that share the pretrained start but use different submissions of that team (a "weight soup"): +$3.3k, 46/48 vs its parent.
  - Averaging different random seeds of the same recipe gave nothing. Averaging fine-tunes on different teams got worse with every extra team.
- **Why pooled corpora fail.** A clone matches its own teacher on 88% of unit actions, but independent top teams agree with each other on only 66–68%. A pooled model takes neighbouring decisions from different players, so its plans stop fitting together. Reactive players clone cleanly; long-horizon planners don't.
- **Architecture** (21.5M parameters, about 16M at play time; the rest is a value tower used only in training).
  - **Tokens:** one per own tile (100), per unit (up to 24), per product, crop and animal, plus a global token.
  - **Opponent encoder:** a separate encoder squeezed into 4 latent vectors, so the opponent can't dominate attention. In practice those latents collapsed to one signal (cosine similarity 0.993): the policy tracks the opponent mostly through the shared market.
  - **Trunk:** 9 transformer blocks. Each head has a learned 19×19 relative-position table indexed by (dx, dy), shared across layers. Removing it lost in paired games.
  - **Draft heads:** a pointer to each unit's target tile plus an action; market gates (open or closed per item and order type), quantities, hires and land.
  - **Movement:** exact shortest-path steps toward the predicted target tile. Every teacher step they measured was a shortest-path move.
  - **Corrector:** 6 self-attention layers over 42 decision tokens.
    - Each unit's soft target distribution becomes an "intent" vector, so two units heading for the same tile show up as a conflict.
    - Market tokens get the expected sells and buys and the shed as it will be at market time.
    - The global token gets expected spend and revenue, computed explicitly rather than learned.
    - It is trained end to end with the loss only on the final decisions, and reads the network's own soft draft. A variant fed with the true plan learned to trust it and lost in live play.
    - **It matters a lot:** it changes 25% of unit targets and the market orders in 74% of turns. Draft alone: about $14k and 0/48 games, against about $100k with the corrector.
  - **Logit adjustment at play time:** subtract the training class priors (β = 1). +$3.2k, 44/48.
  - **Runtime:** plain FP32 PyTorch on 2 CPU threads, about 32 ms per pass. They avoided autoregressive decoding because 15–25 passes per turn would have taken 486–811 ms against the 1 s limit.
- **Rules that close gaps the teacher never visited.**
  - Feed guard: buy exactly the wheat needed for animals not yet fed today.
  - Night-shed sell: simulate the night and sell what would be destroyed (+$0.6k).
  - End sell: sell the whole shed on the last turn.
  - Six other "common sense" rules were neutral or negative. Even blocking requests larger than the shed lost money, because the engine turns such a request into a harmless no-op while a blocked unit wastes its turn.
- **Finding spurious correlations.**
  - One clone had 0.937 validation accuracy and still lost $86k. It had learned "buy land when wheat per animal is low".
  - Found by swapping the teacher's input feature groups into the clone's state one at a time.
  - Fixed by randomly shifting the shed's wheat on 30% of rows and recomputing every feature that shows it: +$45k.
- **Evaluation.**
  - Validation accuracy correlated **negatively** with money (rank correlation −0.86 over 8 versions). The best-ever validation score came from a data leak, and that checkpoint went broke.
  - They selected checkpoints only by money on paired seeds (typically 48).
  - Strength was non-transitive, so in the final week every candidate played a panel of 5–7 opponents.
  - The worst bugs were train/play mismatches, such as computing the shed before vs after the market resolves. They were caught only by step-by-step comparison with the engine.
- **RL** (PPO on change in net worth) gave a few thousand dollars over days of GPU time. Relabelling the data gave more in an afternoon, so both final submissions are pure imitation.
- **Didn't work:**
  - mixed corpora with teacher-style embeddings;
  - routing between expert models per decision head;
  - feeding each unit its previous action (it learned to copy itself: −$25k);
  - all tiles in the corrector;
  - sampling instead of argmax;
  - multi-pass refinement.

### 3.3 7th place: behaviour cloning only (DECEM)
Source: [discussion 745367](https://www.kaggle.com/competitions/kaggriculture/discussion/745367), code at [github.com/Decem-Y/kaggriculture-bc](https://github.com/Decem-Y/kaggriculture-bc).

- **Background.** Seven weeks of heuristics first, then BC in the last 9 days.
- **Data and conditioning.**
  - (state, action) pairs from one seat at a time.
  - Embeddings for team, date bucket (about one per team version) and whether that player won are added to the global token.
  - At play time it asks for the strongest team, its newest bucket and won = 1.
  - Recency weighting with a 3-day half-life. Fresh data mattered more than anything else.
- **Model** (8.2M parameters, 237 tokens per turn, always written from our own seat).
  - **Tokens:**
    - 1 global token: money, hires, quadrants, open shops, day and hour, plus the conditioning.
    - 12 item tokens: price vs base, inventory drift, shop drain, own stock.
    - 24 unit tokens.
    - 200 tile tokens, 100 per farm. Every tile field is a small integer with its own embedding table, summed.
  - **Encoder:** 8 layers, d_model 256, 8 heads, pre-LN.
  - **2-D axial RoPE:** half of each head's rotation pairs turn with the tile's column, half with its row.
  - **Soft cap on all logits:** 30·tanh(x/30).
- **Unit decoder (autoregressive, farmer first).**
  - The query is the unit's token plus the token of the tile under it. It cross-attends to a memory of earlier units' choices, under a causal mask.
  - **Each unit picks four things in order:**
    1. a task (14 types);
    2. a target tile (pointer);
    3. the raw action, conditioned on the task, the target and (Δx, Δy);
    4. an item and quantity where needed.
  - Only the raw action goes to the engine. Task and target exist so later units can see what's already claimed.
- **Market heads** read straight off the encoder. Quantities are classes over bins: every integer up to 30, then coarser steps up to 100 for sells and 1000 for purchases.
- **Labels.**
  - Task = the unit's next non-move action that day; target = the tile it was standing on when it did it.
  - Sells are capped at what's actually in the shed after this turn's unit actions.
  - Animal purchases count animals that actually arrived.
  - Plain cross-entropy, teacher forcing.
- **Decoding details that mattered.**
  - The raw action marginalises over the top-8 (task, target) plans.
  - A legality mask replays the turn so far, unit by unit.
  - Market quantities go through an expected-quantity accumulator. Per-step argmax says "0" every turn when timing is uncertain, so the thing never gets bought.
- **Recipe.**
  1. Pretrain on about 10k games from two weeks (30k steps, batch 512, Muon optimizer, about 50 min on one RTX 5090).
  2. Fine-tune on the latest day.
  3. Specialist fine-tune on the newest versions of one team. It was the only way to learn a new opening.
  4. Evaluate EMA weights.
  - Runs in NumPy, about 0.1 s per turn, 30 MB package.
- **RL on top.**
  - Evolution strategies in the conditioning space (a 16-dim offset to the team embedding): +$1.4k against the clone league, about +$800 on real records. Submitted.
  - PPO on the same offset: about +$1k.
  - Full-network PPO with KL to BC: both players got richer, and win rate didn't move.
  - Per-unit credit assignment: units skipped jobs that didn't pay in their own books, and the opponent gained $2.6k.
  - Using recorded opponents as the RL environment taught exploits that only work because a recording can't react.
  - **Why he stopped:** losses to the top 3 had a median margin of about $4.7k, and only 12–15% were within $2k. A +$1k per game edge flips about one loss in seven.
- **The two rules he kept** (both stop the engine destroying goods):
  - Overflow sell: before night, sell from the shed what the unload won't fit. About +$300.
  - Last-day drop-off: units carrying goods head back and DROP when the remaining steps equal their distance to the shed.
  - On 61 held-out forks they turned 6 losses into wins and no wins into losses.
  - A "rescue watering" rule was +$1.6k on one model and −$2.8k on the next. Rules on top of a learned policy can flip sign.
- **Evaluation.**
  - Community C++ engine port with a parity check for every new opponent.
  - Paired games counted as strict flips (loss→win vs win→loss) with a sign test.
  - Every gain split into "I made more" vs "they made less".
  - **Real-game forks:** replay a real ladder game to day 8, then the candidate takes over while the opponent keeps its recorded actions. Re-running his original agent reproduced 99/99 games to the dollar.
- **Didn't work:**
  - copying single macro moves from other teams (4th quadrant on day 10, extra cows, tomatoes on day 16); each hurt on its own;
  - a contrastive auxiliary loss;
  - classifier-free-guidance-style extrapolation of the conditioning.

### 3.4 10th place: BC then PPO league, small model (MarvinTMB)
Sources: [discussion 745258](https://www.kaggle.com/competitions/kaggriculture/discussion/745258), [discussion 744939](https://www.kaggle.com/competitions/kaggriculture/discussion/744939), code at [github.com/CarsonBurke/kaggriculture](https://github.com/CarsonBurke/kaggriculture).

- **Setup.** An actor of about 947k parameters, trained about 10 h on one RTX 5090: BC on leaderboard games from 27–28 Sep, then a PPO self-play league.
- **Backbone.** An ~800k-parameter [LeJEPA](https://arxiv.org/html/2511.08544v3) backbone sped up learning and let the actor and critic share parameters without gradient conflict.
- **Movement.** A categorical over all 99 destination tiles (masked) beat a categorical over the four directions.
- **Critic.** A win/draw/loss cross-entropy critic beat a bank-margin reward and dense asset-liquidity rewards.
- **League.** Only the toughest opponents. That set keeps changing, so he screened old versions with high posterior σ.
- **Strategy collapse.** The policy ended up buying 3 of 4 land plots every game. An entropy bonus came too late. He'd pretrain the critic on BC data next time.
- **Workflow.** A coding agent first documented the game mechanics, surveyed the public models and data, and built a first model (about 1k rating after an hour of training). Then a few hundred ablations of about 30 minutes each on a Rust environment.

### 3.5 Top 10: supervised fine-tuning, token by token (Russell Kirk)
Source: [discussion 745036](https://www.kaggle.com/competitions/kaggriculture/discussion/745036).

- **Approach.** Treat a turn like a paragraph of text: generate worker commands, then orders, then STOP, one token at a time, each seeing earlier choices that turn.
- **Abandoned.** Two weeks on a macro factoring (311 claims) that proved too much to keep consistent.
- **Data.**
  - **Winners only**: never imitate a losing action. In games we lost, train on the player who beat us.
  - Draws weighted to fresh and strong games.
  - Last day: only the top 3 teams' wins, with seats that beat #1 counted 4×.
  - Short top-ups: 15 min of low-learning-rate SFT on the newest winners.
- **Architecture notes.**
  - Many narrow parallel heads (36 of width 12, regrouped to 18 × 24) did well.
  - Depth over width was worse.
  - Recurrence in depth looked promising but needed too much tuning.
  - Chain-of-thought labels on tiles failed.
- **The field drifts.** An agent of fixed strength scores several hundred points lower later in the competition, especially after many teams moved to four quadrants in the last three days. Ratings only mean something relative to the field at that moment.
- **Workflow: keep 2–3 ideas alive.** Each has a fallback, a current best and a current experiment. Push the experiment until it beats the best in paired games, then the best becomes the fallback. In effect, a tree search over ideas.

### 3.6 31st place (24th on 4 Oct): macro BC plus a deterministic day compiler (t fuku, PavelSavchenkov)
Source: [discussion 745783](https://www.kaggle.com/competitions/kaggriculture/discussion/745783), code at [github.com/PavelSavchenkov/kaggriculture](https://github.com/PavelSavchenkov/kaggriculture/tree/main/final_handoff). Built in about 5 days with Claude Opus 5.5.

- **The split.**
  - Decisions that pay off on later days go to a network, once per day at dawn: what to grow, herd size, land, which crops to keep, and service counts per group.
  - Decisions that pay off today go to an exact solver: tiles, routes, hires, sale timing.
- **DayIntent interface.**
  - Outcomes, not actions: `new_cow = 2` means two cows placed and alive tonight.
  - Counts per group of identical crops or animals, not per tile.
  - Impossible fields are fixed to 0 from the dawn state, with the same mask in labels, training, decoding and compiling.
  - Exact labels: every expert day is replayed in the exact engine and converted. 0 failures in 1.45M days.
- **Training.**
  - Data: 29k episodes (1.45M player-days).
  - **Condition, don't filter.** Conditioning inputs:
    - strength (rating minus that date's ladder reference);
    - recency;
    - style (a one-hot of 9 top teams);
    - each dropped to "unknown" 10% of the time.
  - Filtering to the strongest players lost $11.3k per game. 13× more data with conditioning lifted their local win rate from 79% to 88%.
  - Days 0–5 decoded in one team's style; later days pooled.
  - Logits of 4 older networks averaged in from day 6: about +$1.8k.
  - Seeds with equal validation loss won anywhere from 3 to 23 of 40 games. Select by games.
- **Day compiler** (about 4,000 lines of C++, 1–2 s at dawn).
  - Route construction plus local search for the farmer and up to 13 hires.
  - A hire-count search.
  - Just-in-time purchases, with a funding check that plays the whole plan hour by hour in an engine copy.
  - When money is short, trims one unit at a time, with a survival cash floor for feed.
  - Plans tonight's shed room.
  - A small menu of alternative plans scored as our money + stock − opponent's money.
  - Sales listed by revenue at stake, because earlier slots get better quotes.
  - Slots reserved for dawn sales on heavy hiring days.
- **Hourly seller.** A dynamic program per product over the remaining hours, covering:
  - the price impact of our own units, plus town and shop drains;
  - the opponent's forecast sales, using same-turn quotes exactly as the engine computes them;
  - an objective of our revenue minus the opponent's revenue;
  - stock kept overnight valued at 0.95 × tomorrow morning's expected price;
  - 16 sampled opponent sale paths; ties go to selling now.
- **Opponent sales forecast** (the largest single component).
  - Without it the agent lost 20/20 to itself with it, by $4.2k per game.
  - The opponent's sales aren't visible, so they're inferred every hour: market inventory change + known drains − our actual fills. Using our requested quantities instead of actual fills turns our own failed orders into fake opponent sales.
  - A causal transformer over days predicts the next 24 hours, with an intra-day re-forecast head and a next-morning head.
  - **Timing matters more than totals:** an oracle of the opponent's timing was worth +$8.3k, an oracle of its totals +$4.5k.
- **Validation.**
  - Five test beds:
    - live replays;
    - take-over continuations from any dawn (700/700 days reproduced first);
    - frozen top-team games;
    - teacher days (compile the expert's own intent to separate compiler errors from network errors);
    - a local leaderboard that reproduces official games to the dollar.
  - Sequential gates at 24/48/96/192 games: reject if mean + 1.28 SE < 0, promote if mean − 2 SE > 0.
  - Adopt a change only after it holds on 4 or more seed sets.
  - Beds with real opponent behaviour come first: our own agents sell later than real teams, so self-made beds overrate early-selling changes.
- **Gains came from traced lost games**, each fixing the compiler's model of the game:
  - trim crops before animals on day 0: +$6.6k;
  - trim one unit at a time: 14 → 42 wins of 64;
  - survival cash floor: one world went from −$186k to −$7.9k;
  - sort sales by revenue at stake: 39/40 mirror games;
  - price same-turn sales the way the engine does: +$1.6k to +$4.6k;
  - sell the morning's goods to fund same-day purchases: +$1.5k.
- **Many gains were denial:** the opponent earned less while our own money barely moved. Always read both separately.
- **Infrastructure.** A self-hosted local leaderboard: a PR adds an agent, a Docker worker validates it, plays the top 10 on 15 seeds × 2 seats, and publishes Elo.

### 3.7 About 55th: RL on daily macro plans plus a Rust planner (Snorlax)
Source: [discussion 741743](https://www.kaggle.com/competitions/kaggriculture/discussion/741743).

- **Policy.** It predicts a daily plan per tile: what to grow or keep, and what care is needed. A Rust planner handles hiring, task allocation, movement and sequencing under money, time and capacity limits.
- **Training.**
  - BC on top-10 replays (794k decisions) as a warm start.
  - Then PPO against a mix of replay-based opponents, past checkpoints and self-play: 148 updates × 512 games.
  - Hardware: one RTX 5070 Ti and 16 CPU cores.
  - Reward: bank difference early, later plus win/loss, plus auxiliary rewards.
- **Lessons.**
  - The executor, not the network, was the throughput bottleneck.
  - 95%+ wins against fixed replays carried little signal, because beating a recording isn't beating the adaptive agent.
  - He dismissed turn-level action prediction too early, which turned out to be the strongest approach.
  - RL for commodity trading never worked; trading stayed rule-based.

### 3.8 Other community lessons
- **Syed Asad Ali, 83rd** ([discussion 744939](https://www.kaggle.com/competitions/kaggriculture/discussion/744939)):
  - exact fast simulator first, verified on real replays;
  - rule-based day-1 baseline;
  - imitation, then RL fine-tuning, started early (his regret);
  - hand rules only for measured failures;
  - paired tests with the acceptance rule written beforehand;
  - trust the live ladder over local head-to-heads;
  - plot money by day against the winner's to find where the gap opens;
  - bug-check before abandoning an idea;
  - look-ahead search in spare time per move;
  - verify the exact submitted file plays like the local model.
- **"Just A game on your lips"** ([discussion 744939](https://www.kaggle.com/competitions/kaggriculture/discussion/744939)): a weak BC start, assumed to help RL explore, was a mistake; a strong BC start is better. Two-day RL experiments killed iteration speed.
- **上原 歩夢** ([discussion 743716](https://www.kaggle.com/competitions/kaggriculture/discussion/743716)): RL on a 196-core server reached 90% against public scripts but only about top 100 at best. Against strong players it was weaker than earlier dynamic-programming agents. The training population and objective matter more than compute.
- **Zakaria Aala** ([discussion 744789](https://www.kaggle.com/competitions/kaggriculture/discussion/744789)):
  - Two-level planner idea: a daily LP produces shadow prices per product, tile and labour-hour; each turn greedily assigns units to the most valuable jobs and sells when the live price beats the shadow price.
  - Re-solve daily from the real state (MPC-style).
  - Traps: the model drifts from the real game (walking, night crew resets), plans that are optimal solo lose once the opponent drains the same demand, and a 4th quadrant needs commitment beyond a one-day horizon.
- **Chris Deotte** ([discussion 745788](https://www.kaggle.com/competitions/kaggriculture/discussion/745788)): a solo bot that ignores the opponent can bank $200k+; add opponent adaptation afterwards.
- **Runtime version checks** ([discussion 745445](https://www.kaggle.com/competitions/kaggriculture/discussion/745445)):
  - After the deadline the runtime was relabelled from kaggle-environments 1.32.7 to 1.33.0 for 2–6 Oct (version number only, no game-logic change).
  - Agents with strict version checks crashed every game: one team lost 135 and 150 games.
  - Staff reverted and said they would void those games. Ours were unaffected, because the engine we carry is vendored and doesn't check versions.

### 3.9 Posts from 8–13 Oct

- **Chris Deotte, rank ~439: two bots, and why the schedule bot stalled** ([discussion 747538](https://www.kaggle.com/competitions/kaggriculture/discussion/747538)). One human plus five AI agents, in the last ten days.
  - **Schedule bot ("tape"), final 1960, rank 658.**
    - He rebuilt the public "2945 Farm" notebook, a fixed script that ignores the opponent, and checked it played the same game to the dollar.
    - Then he added three layers, each worth a few hundred dollars a game on 400 paired worlds:
      - a sale-timing layer (sell in the best hours);
      - "hold 8" (from day 20, keep 8 items back to protect prices);
      - a late-game wheat-lot size of 60.
    - It stalled because hundreds of teams copied the same notebook. Tape vs tape was a coin flip: 50% after the deadline, median margin $29.
    - Above about 2100 the opponents were herd-based "economy" bots, not tapes.
  - **Economy bot ("cow bot"), final 2125, rank 439.**
    - It copied the top teams' opening: everything spent on turn 0 (two cows, three sheep, four hands, melon and wheat seeds, feed wheat).
    - Then one measured change at a time, kept only if it won more of the same 400 worlds: **17% → 67% win rate against his own tape in 40 hours.**
    - The changes that mattered:
      - three quadrants, never the fourth;
      - a day-6 herd wave sized by which shops opened (sheep with a yarn store, geese only with an egg shop, cows for milk shops);
      - a day-10 melon rush planted beside the shed;
      - a day-12+ route scheduler with fertiliser reservation;
      - route repair, and helpers swapping route tails;
      - one combined order list on day 0 (worth about 7 points on its own).
  - **"Wheat wash" market trick.**
    - When a rival is detected buying big wheat lots to resell later, buy 60 wheat in the first order slot and sell the same 60 back in the last slot of the same turn.
    - Orders settle slot by slot, so we end where we started and the rival pays a higher price for its lot.
    - It fired in about 1 game in 10 and was worth about $1,000 when it did.
    - The turn-0 "buy 8 / sell 8 / buy 5 / sell 5" pattern in the clones that beat our v16 is likely the same idea.
  - **Post-deadline analysis of 352 games:** economy bots beat both of his bots by $5–8k a game. His moral: "copy the economy, not the schedule, and start on day one."
- **Zakaria Aala, gold zone on free Kaggle GPUs** (comment on 747538):
  - Imitating top replays with a network took him from about 87% to about 102% of the top player's money in that player's own seat.
  - **Better test gate:** replay recorded top players' games with their moves pinned and the market live. Gains measured against his own previous bot often didn't transfer.
  - **Fourth quadrant:** in his close losses the winners averaged 3.46 quadrants to his 3.00. They used the extra one for late tomatoes and wool, and sold about twice his tomatoes (about $5k a game), enough to flip those games. Top players commit to it early, in the games where it pays.
- **MarvinTMB (10th): pure RL can work** ([discussion 745788](https://www.kaggle.com/competitions/kaggriculture/discussion/745788)). BC mainly saved training time: fresh RL runs caught up with his weaker BC runs within an hour or two. What removed his RL plateau:
  - a better model: LeJEPA backbone, `moveto(x, y)` targets with pathfinding instead of move-by-direction, finer market quantities;
  - a win/draw/loss critic instead of dense or soft rewards;
  - warming up the critic on the random model;
  - a hardness-focused league that screens historical opponents with uncertain ratings;
  - enough training time at a lower learning rate.
- **k256.dev, silver in 2 days** ([discussion 748305](https://www.kaggle.com/competitions/kaggriculture/discussion/748305)).
  - They copied a near-top (~106th) submission, choosing one whose play depends only on the town's shop sequence, not on the opponent.
  - With the first three shops there are only 8³ = 512 patterns, so the whole behaviour can be tabled and copied exactly.
  - The rule: **only non-reactive agents are copyable as recordings.** Our failed transplant took reactive top-10 players instead. The author calls it a dirty trick that wasn't enjoyable.
- **Sayaka Miki:** BC was far more effective than RL here, because the game is simple compared with PTCG or Orbit Wars.

### 3.10 Final-week write-ups (14–16 Oct)

The 1st, 4th and 7th place teams have not yet posted their full write-ups.

#### Yuan800: gold range with plain BC, under $60 of compute
Source: [discussion 749586](https://www.kaggle.com/competitions/kaggriculture/discussion/749586).

- **Background.** Rule-based agents for most of the competition (schedulers, route splicing, planner + executor). They stalled because "we could copy what the top teams built, but our executors lost a large share of the throughput doing it."
- **The switch.** BC in the last two days: a Transformer of 4.0M or 7.0M parameters, trained from scratch on one rented RTX 5090 (the larger in about 4.5 h). No self-play, no RL.
- **Why BC worked the second time** (in August it had matched 99% of common actions and still played badly):
  - **The action language must round-trip.** Re-encoding quantities through their converter, with no training at all, turned an expert that won every game into one that lost every game. So they checked that replay → labels → decoded actions is exact before training anything.
  - **Invalid requests are masked, not taught.** A request the engine turns into PASS is removed from the loss, not used as a PASS label.
  - **The whole team is decided together.** One decoder pass predicts every worker's action plus an *ordered* list of market orders, so later orders account for the cash earlier ones use.
- **Inference.** A "shadow resolver" commits orders one at a time against cash, seeds and shed space; an endgame routine brings goods home and sells them.
- **Selecting by games.** The checkpoint with the lowest validation loss finished last in a round robin. Only the final checkpoint beat the incumbent: 92–36 on 64 fresh seeds.
- **RL hurt the deployed greedy policy.**
  - KL-constrained post-training improved the *sampled* policy but made the *argmax* policy worse.
  - A tiny shift flipped one decision (hour 44, SOUTH → PASS) in all 64 games, and the candidate went 25–3–36 against its own base.
  - When you deploy argmax, a small average KL is not a small behaviour change.
- **Weight averaging** of two fine-tuned branches was weaker than the base.
- **A Bradley-Terry lesson.**
  - Final-256 beat Final-192 92–36 locally and won more ladder games overall (543/993 vs 529/1,000), yet rated lower.
  - It lost almost one-sidedly to a few strong teams near its rating (1–57 vs Majkel1337, 7–48 vs Luca), which Final-192 played about evenly.
  - In a Bradley-Terry fit, reliable losses to similar-rated opponents cost more than extra wins against weaker ones.
  - Their local test had only one opponent identity, a near-sibling, so "92–36" measured 256 against 192's habits, not against other styles.

#### Arda Barış Özten: closed-loop per-unit BC on a 12 GB GPU, final 2807.4
Source: [discussion 749669](https://www.kaggle.com/competitions/kaggriculture/discussion/749669).

- **Weeks 1–3: hand-built agents**, which plateaued at 2,550–2,800.
  - An exact C++ port of their agent ran 817 games/s on 12 threads. With it they trained a REINFORCE sale-horizon table over 4.3M games.
  - **The route (what you grow, and when you buy land and animals) is worth far more than the market layer.** Eleven market-layer levers were all refuted.
  - **Mirror games are decided by who sells first.** Any fixed sale horizon loses once copied; a reactive lead detector beat every fixed horizon.
  - An exact cash audit put DSM about +$8.9k per game ahead after controlling for shop luck. Tape executors, DSM-plan executors and a reactor on their planner all failed to close the gap.
- **The switch: closed-loop per-unit BC.** Every turn, every unit picks an action from a network that sees the current state.
  - **Data:** only the cloned teams' seats; every game re-simulated exactly and kept only if money matches (1644/1644). It grew from 1.6k DSM seat-games to 10.3k from the top ~30 teams.
  - **Models:**
    - a residual CNN over the 10×10 farm;
    - then a ViT over 100 tile tokens;
    - finally "worker_hybrid": two local convolutions, then attention over tile tokens plus explicit worker tokens (width 256, 6 layers).
  - **Heads:**
    - unit head: 47 action tokens;
    - market head: 21 order keys × 18 count buckets, where bucket 0 means no order;
    - label smoothing 0.02.
  - **Decoding** is parallel, one trunk pass per turn, not autoregressive.
  - **Runtime:** NumPy with fp16 weights base64-encoded into a 5.6 MB `main.py`; 20 ms (CNN) to 73 ms (ViT) per turn.
- **The decoder mattered more than the network.** The first trained build finished on $0. The fixes:
  - **calibrated per-key market thresholds** (argmax never fires rare orders like `BUY_LAND` or `HIRE`), set so the clone's held-out firing rate matches the teacher's;
  - **a sell-what-arrives clamp:** SELL counts clamped to shed stock *plus* goods placed or dropped this turn, since DSM sells the turn it delivers;
  - a legal-op mask at inference only, a PLANT cap at seeds in stock, and a last-day dump.

  The same weights went from $0.1k to $75.6k. Later the clone made more than DSM itself in DSM's hold-out worlds ($128k vs $121k).
- **Architecture sweep.** Spatial attention beat temporal context by a wide margin; nearly all the gain from history came from the previous action alone.
- **Evaluation.**
  - **GPU-batched gauntlet:** one forward pass serves every waiting game; 48 games in 101 s instead of 13 min.
  - **Pinned ladder worlds** (their real losses with the rival's recorded actions fixed) were a screen; closed-loop head-to-head was the gate. One fine-tune scored +$24k per game pinned, but 54 of its pinned rival tapes crashed once the market diverged.
  - **Count seeds, not games.** Between near-identical clones, about 70% of seat-swapped seeds produce mirrored, identical results, so 24 games are about 12 independent worlds.
  - **Gate rule:** the candidate must win the incumbent's ladder losses and beat the incumbent head-to-head.
- **The tomato gap.**
  - Top teams started tomatoes late (11.5 plants by day 20) and sold them at about $113. Their DSM teacher didn't grow tomato.
  - A wrapper (swap `PLANT WHEAT` → `PLANT TOMATO` when the opponent grows tomatoes on days 13–20) gave +$2.2k, with 7 losses flipped and 0 wins lost. Early tomato (day 8) starved cash and lost $6–9k.
  - Fine-tuning on DSM's newest tomato-always versions (219 seat-games, 3.5 minutes) added +$2.4k more.
  - **A crop strategy has to come from the teacher, not a wrapper.**
- **Didn't work:**
  - **Unit-coordination masks** (dedupe units claiming the same task): **−$42k to −$89k per game.** The clone had learned DSM's own collisions, and "fixing" them pushed it off-distribution.
  - single-team fine-tunes on other teams;
  - mixing the top 20 with a style slot;
  - weight soup;
  - a 3-model logit ensemble (+$0.5k but no extra wins, 3× slower);
  - head-only PPO (tie).

#### KevlarZanderChi: exact-search distillation and VinePPO on a public base, about rank 1,140
Source: [discussion 749460](https://www.kaggle.com/competitions/kaggriculture/discussion/749460).

- **Setup.** A ~40k-parameter RL layer on the public Pipe-16 agent makes four stage decisions per game (days 0, 8, 15, 23), choosing among 8 strategy options.
- **Why plain PPO/GRPO stalled: signal vs noise.**
  - A stage decision moves the final margin by a median of **$42**; seed-to-seed noise is about **$665** (SNR 0.06). Learning one decision's sign from sampled returns needs about 1,000 games.
  - Only 5.6% of decisions have any option that flips the result, and that can't be predicted from the features.
- **What worked: remove the noise instead of averaging it.**
  - `os.fork()` the exact simulator at each decision, including both agents' memory.
  - Play all 8 options to the end against the live opponent.
  - Train first by search distillation, then **VinePPO** (advantages from exact forked rollouts).
  - Result: 14 better / 7 worse of 150 paired games, and about +100 rating live.
- **Replay worlds.** The opponent's recorded actions on the recorded seed reproduced 66/66 of his ladder losses: a test set made of the real ladder.
- **The carry-trade trap.**
  - Wheat buy-and-resell options won close mirrors but collapsed against an older agent (15/15 → 10/15).
  - **Test against opponents your training never saw**, and gate on "no collapse against anyone", not net wins.
- **Copying top-10 moves into a public base improved 0 of ~500 paired games.** Top-10 behaviour comes from a planner built around it, not a list of moves. This is the same finding as our own and DECEM's.
- **The bronze line was a Pipe-16 mirror.** 66% of his ladder games were decided by under $500.
- **Lesson:** measure effect size against noise before choosing an RL method.

#### shawnYiB: replay schedule plus a forecast-guided seller, 2226.6
Source: [discussion 749473](https://www.kaggle.com/competitions/kaggriculture/discussion/749473).

- **The question that changed things:** not "how do I make more money?" but "how does this decision change the money difference between my opponent and me?"
- **Freeze the farm to isolate the market.** With production fixed, an opponent-aware revenue search lifted win rate from 62.5% to 89.6% (margin +$3.2k: our cash +$0.5k, the opponent's −$2.8k). Against same-production opponents it went from 45.8% to 100%.
- **An explicit penalty on rival income was worse** than modelling the rival's supply and scoring our own revenue conservatively. The final penalty weight is zero.
- **Opponent-sales forecaster.**
  - Small gradient-boosted trees with a Poisson loss predict rival sale quantities over [0,1), [1,4), [4,12) and [12,24) turns.
  - 16 economic features: time, market and demand, visible rival capacity, inferred supply history.
  - Labels are executed sales, not requested orders. The trees run in pure Python.
- **Seller search.**
  - Candidate quantities: none, ¼, ½, ¾, all, and the rule-based amount.
  - Horizons: the known shop-consumption ticks within 24 turns.
  - Scenarios: 3 rival volume scales (0.5/1/1.5×) × 3 settlement positions (before, alongside, after the rival).
  - Score: expected revenue − 0.2 × downside − a holding penalty.
  - It defers to the rule-based seller when cash is under $10k, and for wheat, fertilizer and the endgame.
- **Production changes are small plan recompilations, not text substitutions.** Changing what a worker plants also changes what to buy, harvest, carry and deliver.
- **Daily-target imitation is a trap.** Tile-state accuracy looks strong because most tiles don't change, and it hides the few critical errors. "The learned representation and the executor must agree about what can actually be executed."

#### Masahiro Yamagata: 347th (silver), CPU only, with an autonomous Claude improvement loop
Source: [discussion 749466](https://www.kaggle.com/competitions/kaggriculture/discussion/749466).

- **Strength is mostly early investment.** Across 490 top-5% submissions (26,424 games), investment through day 10 has a rank correlation of **0.75** with strength.
- **Pick one type that works, don't average types.** He chose the four-quadrant type: it appeared on 22 Sep and took 61% of top seats within three days.
- **Targets.** From 22 reference submissions that buy the 4th quadrant on day 10, he set per-day targets at two levels (reached in 90% and 50% of their games), conditioned on open shops. Example: cows on day 10 are 4 with no milk shop, 14 with three.
- **Measuring.** Start from the reference's recorded games and play only that day's moves. A fix that lowers a 90%-level target is rejected even if it ends with more money.
- **The loop.** A coordinator agent opened a ticket per unmet target, and child agents analysed, fixed and measured. Two days produced 169 tickets, 4,407 experiments and 509 adopted fixes, and took him from rank 3,571 to 286.
- **Remaining gap: sell timing**, which wasn't in the targets. BC on his Mac had matched 84% of actions and still ended 67% short in money.

#### Heuristic portfolios: 202nd and 512th
- **yash bhaskar, 202nd, 2345.5** ([discussion 749461](https://www.kaggle.com/competitions/kaggriculture/discussion/749461)).
  - A portfolio of complete, turn-indexed schedules extracted from public replays, chosen by the first two shops (visible at steps 72 and 144):
    - milk-oriented if both consume milk;
    - wool-oriented if a yarn store appears;
    - otherwise general.
  - Safeguards:
    - keep 4 coins late on day 1 for the next morning's hires;
    - turn excess planting requests into PASS when a seed purchase was skipped.
  - Weakness: switching schedules mid-game without reconciling the farm with the new schedule's expected state.
- **Sarthak Sharma, 512th, bronze; peaked at rank 13 with 2810** ([discussion 749453](https://www.kaggle.com/competitions/kaggriculture/discussion/749453)).
  - A shop-conditioned route portfolio plus reactive controllers for capital, market and execution. BC failed for him because errors compounded off-distribution.
  - Lessons:
    - find the *first* decision that created the gap;
    - test every replacement directly against the incumbent (a "fix" for the diagnosed cause lost badly);
    - promote by paired-seat wins, not margin;
    - recorded-opponent panels (27–3) are diagnostics, not live proof.

#### Follow-ups
- **Chris Deotte** used the same pinned-move gate, which his agents called "ghost bots". He agreed that choosing *which* teams are copyable matters more than copying everyone: copying the whole leaderboard produced an average bot.

### 3.11 What the published code shows (1st, 5th and 11th)

The three repos, as read for this comparison:

| Repo | Licence |
|---|---|
| `msdsm_kaggriculture-solution` (1st) | no licence file: read only |
| `Decem-Y_kaggriculture-bc` (5th) | MIT |
| `CarsonBurke_kaggriculture` (11th) | MIT |

**All three:**
- run on an **exact native engine** (two Rust ports, plus the community C++ port used by DECEM), with the official `kaggle-environments==1.32.7` as the parity oracle and final evaluator;
- use an **entity Transformer** over tile, unit, item and global tokens with **2-D RoPE**;
- predict **market quantities as classes**.

| | 1st: M & M & P & Q | 5th: DECEM | 11th: MarvinTMB |
|---|---|---|---|
| **Stack** | JAX model, Rust batch env, C++ search planner | PyTorch training, NumPy inference (no torch in the package) | PyTorch, Rust batch env (PyO3) |
| **Size** | 12 blocks × 256 wide, 8 heads, FFN 1024; 10.23M params | 8 layers × 256, 8 heads, FFN 1024; 8.2M | width 96, 4 query / 2 KV heads; ~0.95M |
| **Position** | selective 2-D RoPE (rotary dim 16, base 100) | 2-D axial RoPE (half the pairs by column, half by row) | axial 2-D RoPE inside two per-farm blocks |
| **Tokens** | 264: 200 tiles (both farms), 40 units (both players), global, 12 items, **an opponent-inventory memory token**, 10 market-slot tokens | 237: global (+ team / date / won embeddings), 12 items, 24 units, 200 tiles | 200 tiles + 20 economy tokens as source memory; 26 decision states (16 units + 10 market slots) refined over 4 reasoning rounds |
| **Unit actions** | 500 logits per unit | per unit: task (14) → target-tile pointer → raw action (18) → item / quantity, decoded in turn with a causal memory of earlier units | 68 primitives, quantity-bearing pickups; destination scores turned into the 4 move probabilities along a vertical-first route |
| **Market** | 10 ordered slots; 1,003 non-sell choices + 9 products × 100 absolute sell quantities; sequential decoding reserves stock | per-product bins read off item tokens; expected-quantity accumulators | up to 10 ordered slots of 22 kinds incl. STOP; quantities 1–100 + ALL; a per-order resource ledger (29 features) conditions each next order |
| **Labels** | executed actions only; sells use the **executed** quantity | task = first non-move action until day end; sells capped at the shed after this turn's unit actions; animals counted on arrival | replays where the official engine reproduces the game exactly |
| **Training** | 5 BC rounds alternating with PPO: W/D/L ±1 reward, γ 1, λ 0.97, clip 0.2, KL(teacher‖policy) 0.2 | Muon for hidden matrices + AdamW; EMA 0.9998; batch 512; 30k steps; bad-batch and gradient-spike skipping | BC, then PPO with a W/D/L critic; the critic trains alone until R² ≥ 0.10; DAPO clip [0.8, 1.28]; KL target 0.03 |
| **Data** | final BC: 849 public + 600 heuristic self-play trajectories (100 standard, 100 tomato-shop, 100 skewed-shop worlds) | ~10k games, 3-day recency half-life; drops open-loop tapes (byte-identical action streams) | hosted games between agents rated ≥ 2600 |
| **Opponent** | rule-updated estimate of the opponent's private inventory from public observations | the opponent's farm as tokens | a centralized critic sees the opponent's private state (training only) |
| **Inference rules** | collision repair; seed caps; night overflow sell; liquidate on actions 717–718; agent A hands day 29 to a C++ search (~6 s) | overflow sell; last-day return-and-drop | legality masks follow earlier units' effects in engine order |
| **Evaluation** | seat-swapped local games | day-8 forks of recorded games with pinned shops; flip counts, sign test, our-vs-their money split | league with a hardness selector; 5 held-out reference agents never trained against |

**Details worth carrying forward:**
- **Model growth.** Net2Net depth growth (6 → 12 → 24 blocks, new blocks zero-initialised so the policy is unchanged) let 1st place keep training as models grew.
- **Inventory order matters.** MarvinTMB's unit features include each item's insertion rank in the bag, because DROP fills the shed in insertion order and discards the overflow.
- **Execution order matters too.** Unit slots follow engine execution order (farmer first) in all three.
- **The final-turn liquidation window is two actions (717–718),** and every repo handles it explicitly.
- **DECEM's census drops open-loop tapes,** teams replaying a byte-identical action stream in several episodes. Their states don't reflect a policy, so they are bad BC data.

---

## 4. Patterns across the leaders

### Policy representation
- **Learn at the turn level.**
  - Every top-10 entry predicted per-turn unit actions and market orders.
  - Daily macro plus a planner reached about 24th–55th.
  - Public heuristic forks ended in the 1,600–1,900 rating band.
- **Coordinate units within a turn.** Three ways worked:
  - autoregressive unit decoding with a memory of claimed tasks and targets (7th);
  - a one-pass draft followed by a self-attention corrector (4th);
  - token-by-token generation (top-10 SFT).
- **Predict target tiles, not steps.** Predict where a unit goes (a pointer over tiles) and let an exact shortest-path rule take the steps (4th, 7th, 10th).
- **Encode space relatively.** Use a learned relative-position bias (dx, dy) or 2-D axial RoPE.
- **Represent the opponent cheaply.** A bottleneck is enough: policies track the opponent mainly through the shared market.
- **Market quantities as classes over bins**, with gates per item and order type, an expected-quantity accumulator across steps, and logit adjustment for class priors.
- **Audit the decoder before judging the model.** Three fixes took one clone from $0 to beating its teacher (Arda):
  - calibrated per-key thresholds for rare orders (land, hires);
  - a same-turn sell clamp that counts goods delivered this turn;
  - a resolver that commits orders one at a time against cash, seeds and shed (Yuan800).
- **Check the action language round-trips exactly** (replay → labels → decoded actions) before training. A lossy converter alone turned an unbeatable expert into a loser (Yuan800).
- **Don't hand-"fix" the teacher's quirks at play time.** Masking unit collisions cost $42k–$89k per game, because the clone left its training distribution (Arda). A learned corrector is fine (4th); a hand mask is not.

### Data and training
- **Clean labels to what the engine executed.** Replay every game and keep executed trades only (4th; 7th capped sells to shed stock).
- **Condition on team, date, strength and outcome rather than filtering** (7th, 24th). At play time, ask for "strongest team, newest version, won".
- **Weight recent games.** The field changes daily (3-day half-life at 7th).
- **Use a staged schedule:** broad pretraining, then a specialist fine-tune on one consistent team. Pooling many teachers produces incoherent plans.
- **Average weights only between siblings** (same pretraining, different fine-tune data). Averaging logits of older networks also helped (+$1.8k at 24th).
- **Imitate winners only** (top-10 SFT). Add the games that beat you.
- **Break spurious correlations** with targeted feature perturbation, found by swapping input feature groups.

### RL: what helped and what didn't
- **What helped:**
  - Start from a strong BC policy.
  - PPO with very large self-play (1st: 8.29M games) or a curated league (10th).
  - A win/draw/loss critic, and pre-training the critic.
  - Small moves such as ES or PPO in the conditioning space (7th, about +$1k).
- **What failed:**
  - Per-unit credit assignment, because units skip team-useful work.
  - Recorded opponents as the environment, because the policy exploits non-reactive tapes.
  - Full-network PPO without a good league, because both players get richer and win rate doesn't move.
  - Weak BC starts.
- **Rule of thumb:** with modest compute, better data (relabelling, conditioning, freshness) beat RL (4th, 7th). Yuan800 reached the gold range with plain BC for under $60.
- **For a greedy (argmax) policy, a small KL is not a small change.** One flipped decision cost a whole 64-game set (Yuan800).
- **Measure effect size against noise first.** When a decision moves the margin by $42 against $665 of seed noise, sampled-return PPO can't learn it. Fork the exact simulator and compare all options in the same state: search distillation, then VinePPO (KevlarZanderChi).

### Rules on top of a learned policy
- **Keep only rules that stop the engine destroying value:**
  - overflow sell before night;
  - last-day drop-off;
  - end-of-game sell;
  - feed guard so animals don't run away.
- **Most "common sense" rules were neutral or negative**, and some flipped sign between weight versions.
- **Build the rules into training if possible.** Rule-aware PPO with action masks (1st, Final B) beat rules added only at inference (Final A).

### Market microstructure
- **Sell order matters.** Both players' i-th units get the same quote and earlier slots get better prices, so sort sales by revenue at stake.
- **The objective is our money minus the opponent's.** Selling first takes price from the opponent. Buying livestock takes away milk and wool demand that is finite and shared.
- **Forecast the opponent's sale timing.** That was the largest single component at 24th; timing was worth roughly twice the totals.
- **The engine's silent clipping and no-ops are often cheaper than enforcing legality yourself** (4th).
- **Same-turn "wash" orders** (buy first, sell back last) cost us nothing but raise the price a rival pays for a big lot in that turn (Chris Deotte: about $1,000 when it fires).

### Copying another agent
- **Copy a policy, not a trajectory.** Recordings of reactive agents fail on new boards.
- **A recording works only for a non-reactive agent**, one whose play depends only on the board or shop sequence. You can then table its behaviour by shop sequence (k256.dev; also the yhay81 shop-router tapes in our base).
- **Copy the economy, not the schedule.** Herd-based economy bots beat fixed crop schedules by $5–8k a game (Chris Deotte).

### Evaluation
- **Exact fast engine.**
  - C++ or Rust port, verified on real replays, with a parity check against the official engine for every new opponent.
  - Your own ladder replays should reproduce to the dollar (99/99 at 7th, 700/700 days at 24th).
- **Paired seeds, both seats.**
  - Count strict flips with a sign test.
  - Sequential gates: reject if mean + 1.28 SE < 0, promote if mean − 2 SE > 0.
  - Require 4 or more seed sets before adopting a change.
- **Test against real opponent behaviour.** Fork real ladder games at day d, or take over a recorded game from any dawn. The opponent keeps its recorded actions.
- **Measure yourself in the top players' own seats.** Pin their moves and keep the market live, then report your money as a share of what the top player made in that seat. Gains against your own previous bot often don't transfer (Zakaria Aala).
- **Use a panel of 5–7 opponents.** Strength is non-transitive.
- **Select checkpoints by games, never by validation loss or accuracy.** The rank correlation was −0.86 at 4th, and equal-loss seeds won 3 to 23 of 40 at 24th.
- **Split every gain** into our money vs the opponent's money.
- **Count seeds, not games.** Seat-swapped games between near-identical agents often mirror exactly (about 70% of seeds at Arda's), so n games can be about n/2 independent worlds.
- **Check final candidates against the teams around your rating.** In a Bradley-Terry fit, one-sided losses to similar-rated teams outweigh extra wins elsewhere. A 92–36 local winner rated lower on the ladder for exactly this reason (Yuan800).
- **Include opponents your training never saw**, and gate on "no collapse against anyone" (KevlarZanderChi's carry-trade trap).
- **Use pinned worlds as a screen and closed-loop head-to-head as the gate.** Absolute pinned scores are biased once the market diverges (Arda).

---

## 5. Our own learnings

**What we built:**
- Strongest public heuristic base agents: the harvest-ledger / master-engine lineage, hl0927 and hl0929.
- An in-game opponent identification layer:
  - Shadow each hypothesis opponent agent through the vendored official engine every turn.
  - Drop hypotheses whose predicted actions don't match the observation.
  - Exploit identified opponents' same-turn market orders (maximin).
- One-knob parameter sweeps on the base, and a turn-0 anti-identification order (v16).
- Paired-seed league testing, fork forensics on lost ladder games, and agent-log reading.

**What worked:**
- **Opponent identification and exploitation** won +$130–400 margins against identified opponents, and every identified game in our samples was a win. It kept us above bronze from 24 to 29 Sep while public clones dominated the ladder.
- **Paired, multi-seed comparison** was the only reliable signal. The board reshuffles whenever behaviour changes, so single games mean nothing.
- **Building on the strongest public base and patching only measured, near-Pareto-safe edges.** One example is `_SR_MARGIN` 28 (night-overflow selling), which won tied mirror games.
- **Diagnostics**:
  - reading Kaggle agent logs (`/episodes/{id}/agents/{idx}/logs`);
  - fork forensics: replay the opponent's observations through candidate agents to find the first step where it deviates;
  - re-simulating recorded games exactly.

**What failed, and why:**
- **Our local leagues stopped predicting the ladder.** They said 92–96% wins while the ladder said 40%. By the end the ladder band was full of private clones with micro-tweaks we didn't have. Lesson: rebuild the test bed from real ladder games (forks and continuations), not from public agents.
- **The exploit layer hedged when near-duplicate hypotheses survived together.** Keep the hypothesis pool free of near-copies of the most common opponent.
- **Bolting macro moves onto a heuristic base:**
  - a south-east goose farm: −$20k margin, 0/24 wins;
  - a relaxed tomato gate: −18.5 pp;
  - copying the top players' 4th quadrant.

  This matches DECEM: single copied macro moves hurt, because the rest of the plan doesn't adapt to them.
- **Replaying top players' recorded games ("tape transplant") on new boards: 0/196 wins.** A recorded game is closed-loop only on its own seed; on a new board one missed sale cascades into bankruptcy. The leaders cloned a *policy* that re-decides every turn.
- **We never built a learned policy.** We stayed on heuristics for the whole competition. Strong BC models appeared in the last two weeks, and the field strength jumped past what heuristic forks could reach.
- **Late-wave drift.** In the final 48 hours thousands of submissions arrived. Every fixed-strength agent lost 100–350 rating points as the field improved.

**Operational lessons:**
- **Slots.** Only the last two uploads count, and each new one retires the older active one. Sequence uploads deliberately; a fresh upload re-runs its warm-up climb.
- **Kaggle loads the last callable** in the file. Appended layers must keep the entry function last.
- **Turn-0 time includes loading.** Measure the first move on Kaggle-like hardware (we saw 0.9–1.4 s on Kaggle against about 0.3 s locally).
- **Don't add strict runtime-version checks.** Vendor what you need instead.
- **Schedulers die with the session.** Our in-session cron checks died when the session ended at the deadline. Use durable scheduling for anything critical near a deadline.

### Post-competition: applying the learnings (16 Oct)

We rebuilt the agent the way the leaders did. The code is in this folder (see the [README](../README.md)).

**Approach:**
- the exact official engine;
- 6,444 games of the top teams' fixed final agents (1–13 Oct);
- labels taken from what the engine executed;
- DECEM's entity Transformer, conditioned on team;
- 3.8M parameters, 3.9 h on the laptop's RTX 3050.

**Results:**
- **Simulation:** all 6,444 games reproduce exactly at every step.
- **In the top teams' own held-out seats** (opponent pinned), the agent made 100–103% of each team's real money:

  | Team | Our money ÷ theirs | Our win rate | Their win rate |
  |---|---|---|---|
  | M & M & P & Q | 1.026 | 50% | 67% |
  | DSM | 1.001 | 62% | 53% |
  | Victor | 1.020 | 55% | 41% |
  | DECEM | 1.008 | 72% | 48% |

  Our competition agent v16 made 91% in the M & M & P & Q seats.
- **Closed loop vs v16:** it won every seed, by +$28k to +$36k per game.
- **Round robin by copied style:** M & M & P & Q 69% average, Victor 65%, DSM 46%, DECEM 21%, the same order as the real leaderboard.
- **Specialising** on one team added nothing over conditioning on it.

**Lesson:** the gap between our heuristic agent (rank ~1,430) and the gold zone was a copied, re-planning policy plus
exact executed labels. That fits on a 4 GB laptop GPU in an afternoon once the data pipeline exists. Building
that pipeline in week 1 would have changed the competition for us.

---

## 6. Do's and don'ts

### Do
1. **Build or adopt an exact, fast engine port in week 1** and verify it reproduces real replays to the dollar. Everything else depends on it.
2. **Set up a replay pipeline early.** Use the daily top-replay dumps, the episodes API, and the replay API for your own games.
3. **Ship a rule-based baseline on day 1**, then start behaviour cloning from top replays as soon as they exist.
4. **Clean imitation labels to what the engine executed.** Recompute features from the same replayed record.
5. **Condition on team, version, strength and outcome; weight by recency; fine-tune on one consistent teacher.**
6. **Make the turn a coordinated plan:** autoregressive units with claims, or a draft plus corrector. Predict targets and use exact pathing.
7. **Select by paired-seed games**, both seats, with sequential statistical gates, a 5–7 opponent panel, and gains split into ours vs theirs.
8. **Test on real opponent behaviour:** forks or take-overs of real ladder games, after confirming your unchanged agent reproduces them.
9. **Add rules only where the engine destroys value** (overflow, last day, end sell, feed), and re-check them whenever the weights change.
10. **Model the market as a two-player resource:** order-slot priority, same-turn quotes, opponent sale timing, denial.
11. **Keep 2–3 lines of work alive**, each with a fallback, a current best and an experiment.
12. **Trust the live ladder over local tests.** Watch how ratings drift across the field, and plan for a strong late wave.
13. **Bug-check a failed experiment before abandoning it.**
14. **Verify the action-label round trip, and audit the decoder** (rare-order thresholds, same-turn sell clamp, resolver against cash and shed) before blaming the network.
15. **Before picking the final two, check each candidate against the teams near your rating** and the opponents your current version struggles with.
16. **Measure a decision's effect size against seed noise** before choosing an RL method. If the signal is tiny, use exact forked counterfactuals.

### Don't
1. **Don't select checkpoints by validation loss or accuracy.**
2. **Don't pool many teachers without conditioning**, and don't average weights across different teachers.
3. **Don't train RL against recorded or non-reactive opponents**, and don't expect "95% vs fixed replays" to transfer.
4. **Don't start RL from a weak BC policy**, and don't make experiments take days. Prove the signal small first.
5. **Don't bolt single macro moves** (land, crops, herd) onto a policy that can't re-plan around them.
6. **Don't replay recorded action sequences on new boards.** Clone policies, not trajectories.
7. **Don't trust a league of public agents** once the ladder is dominated by private forks.
8. **Don't enforce "legality" that the engine already handles as a harmless no-op.**
9. **Don't hard-fail on environment version checks** in a submitted agent.
10. **Don't leave final uploads to the last minutes.** Late uploads were disabled; one team's AI assistant delayed its upload past the deadline.
11. **Don't hand-mask a clone's teacher quirks at play time.** Change behaviour in training instead.
12. **Don't pick a final submission from a test against a single sibling opponent.**
13. **Don't add a crop or strategy through a wrapper when you can change the teacher.** The wrapper hits a local optimum (Arda: ±$230), and switching the teacher added +$2.4k more.

---

## 7. Playbook for the next simulation competition

1. **Week 1:**
   - Write a mechanics document from the engine source.
   - Build an exact fast port and parity test.
   - Build a replay downloader and parser.
   - Build a paired-seed harness with sign tests and SE gates.
2. **Week 2:**
   - Submit a rule-based baseline.
   - Build the "fork a real game" test bed.
   - Read the top replays: money by day against the winners, and when they buy land.
3. **Weeks 3–4:**
   - Behaviour cloning with executed-action labels.
   - Entity Transformer (tiles, units, items, global), relative 2-D position, pointer targets plus pathing, market quantity bins.
   - Condition on team, date and outcome.
4. **Then, depending on compute:**
   - Specialist fine-tunes and sibling weight soups (cheap, reliable).
   - Or PPO from the strong BC policy against a curated, fresh league, with a win/draw/loss critic and rule-aware masks (expensive; 1st place used 8.29M games).
5. **Throughout:**
   - Fold the newest top replays in daily.
   - Trace every lost ladder game to its root cause.
   - Keep the last two submission slots deliberate.

---

## 8. Sources

| Topic | Discussion |
|---|---|
| 1st place preview | [745073](https://www.kaggle.com/competitions/kaggriculture/discussion/745073) |
| 4th place preview | [745119](https://www.kaggle.com/competitions/kaggriculture/discussion/745119) |
| 7th place preview | [745367](https://www.kaggle.com/competitions/kaggriculture/discussion/745367) |
| 10th place preview | [745258](https://www.kaggle.com/competitions/kaggriculture/discussion/745258) |
| SFT write-up, top 10 | [745036](https://www.kaggle.com/competitions/kaggriculture/discussion/745036) |
| 24th place write-up | [745783](https://www.kaggle.com/competitions/kaggriculture/discussion/745783) |
| RL on macro plans, ~55th | [741743](https://www.kaggle.com/competitions/kaggriculture/discussion/741743) |
| "How do top teams approach this?" | [744939](https://www.kaggle.com/competitions/kaggriculture/discussion/744939) |
| RL on a 196-core server | [743716](https://www.kaggle.com/competitions/kaggriculture/discussion/743716) |
| Post-competition reflections | [744789](https://www.kaggle.com/competitions/kaggriculture/discussion/744789) |
| How early public strategies were found | [745788](https://www.kaggle.com/competitions/kaggriculture/discussion/745788) |
| Runtime version change | [745445](https://www.kaggle.com/competitions/kaggriculture/discussion/745445) |
| Final evaluation | [731587](https://www.kaggle.com/competitions/kaggriculture/discussion/731587) |
| Chris Deotte, schedule bot vs economy bot, rank ~439 | [747538](https://www.kaggle.com/competitions/kaggriculture/discussion/747538) |
| Silver in 2 days by copying a non-reactive agent | [748305](https://www.kaggle.com/competitions/kaggriculture/discussion/748305) |
| Gold range with plain BC for under $60 (Yuan800) | [749586](https://www.kaggle.com/competitions/kaggriculture/discussion/749586) |
| Closed-loop per-unit BC on a 12 GB GPU (Arda Barış Özten) | [749669](https://www.kaggle.com/competitions/kaggriculture/discussion/749669) |
| Exact-search distillation and VinePPO (KevlarZanderChi) | [749460](https://www.kaggle.com/competitions/kaggriculture/discussion/749460) |
| Forecast-guided seller (shawnYiB) | [749473](https://www.kaggle.com/competitions/kaggriculture/discussion/749473) |
| Autonomous Claude improvement loop, 347th (Masahiro Yamagata) | [749466](https://www.kaggle.com/competitions/kaggriculture/discussion/749466) |
| Shop-based schedule portfolio, 202nd (yash bhaskar) | [749461](https://www.kaggle.com/competitions/kaggriculture/discussion/749461) |
| Route portfolio, 512th (Sarthak Sharma) | [749453](https://www.kaggle.com/competitions/kaggriculture/discussion/749453) |
| 1st place code | [msdsm/kaggriculture-solution](https://github.com/msdsm/kaggriculture-solution) |
| 5th place code (DECEM) | [Decem-Y/kaggriculture-bc](https://github.com/Decem-Y/kaggriculture-bc) |
| 11th place code (MarvinTMB) | [CarsonBurke/kaggriculture](https://github.com/CarsonBurke/kaggriculture) |

The code for the post-competition rebuild is in this folder; our in-competition scripts (v12–v17) are not published.
