"""A learned policy for the decisions the planner makes with fixed constants.

The planner (:mod:`fplass.optimise`) is a mixed-integer program over an eight-week horizon. It
is very good at the combinatorics — which fifteen, which eleven, which swap — and it handles the
sequential questions with constants: a banked transfer is worth three points, a chip is played
when its gain beats a measured threshold for the weeks left. Those are the questions this
package asks a reinforcement learner instead:

* how many transfers to make this week — none (roll), one, two or three, hits included;
* which chip, if any — Bench Boost and Triple Captain alongside the transfers, or a Wildcard
  or Free Hit instead of them.

The learner chooses; something else executes. Two executors share one interface:

* :mod:`fplass.rl.env` — a fast one, for training: the best disjoint single swaps by horizon
  projection, the best eleven by the week's projection, a one-week solve for a Wildcard or Free
  Hit squad. A season is played in a second or two, so thousands can be.
* :mod:`fplass.rl.replay` — the planner's own solver told how many moves to make and which chip,
  scored exactly as :mod:`fplass.backtest.manager` scores the planner. This is the number that
  decides whether the learner earns a place: nine seasons, each held out from the policy that
  plays it, paired against the planner's.

Learning is fitted Q-iteration (:mod:`fplass.rl.fqi`) with gradient-boosted trees, on transitions
collected by randomised versions of a rule-based policy and by a policy cloned from the top
managers' logged decisions (:mod:`fplass.rl.elite`). Batch learning suits the problem: a season is
37 decisions, the reward is noisy, and a tree model of the value of each action given a few
dozen state features needs neither a neural network nor millions of episodes.
"""
