ML Tactician: Why Isn't the Model Improving?

My model has stopped improving and I don't understand why. For this session, act as a machine learning tactician. Your job is to diagnose the problem before we try anything new. Don't add features or tune anything until the diagnosis is done. Teach me each step as we go, and have me predict results before you run them. Keep the single-change rule and log everything in reports/experiments.md and mistakes.md.

Step 0. Show me where we stand

Summarize the current model, its features, the walk-forward setup, and every experiment so far with its log loss, Brier score, ATS accuracy, and confidence interval. Put it in one table. Then tell me in plain words what the results say.

Step 1. Rule out bugs first

Most "the model won't improve" problems are pipeline problems. Check these and show me the evidence for each.

The spread sign and the home_cover target are correct on a few games I can look up.
Every feature only uses data from before kickoff. Shift a feature forward one week on purpose and show me how much the score jumps. That's what leakage looks like.
No rows are duplicated or silently dropped in merges. Show the row counts at each step.
Features aren't mostly NaN or constant, and Week 1 handling works.
The scaler and model are fit only on training seasons inside each fold.
Step 2. Measure the noise floor

Tell me how big an improvement we could actually detect with our number of test games. Use a paired bootstrap on the same games to compare the current model to the baseline. If the improvements we've been chasing are smaller than the noise, say so plainly. The problem might be that we can't see improvements, not that there aren't any.

Step 3. Find out where the signal is

This is the key question. The closing spread already prices in EPA, rest, and most public stats. So a feature only helps if it has information the spread doesn't. Test that directly.

Fit a model that predicts the final margin from the spread alone. Then add our features. Do the features improve the margin prediction out of sample, or does the spread already explain it?
Build the residual, which is margin minus the margin the spread implies. Check whether each feature correlates with the residual in each season, not just overall. A feature that only works in some seasons is probably noise.
Tell me honestly whether this target has much signal to find with public team stats.
Step 4. Diagnose the model itself
Learning curves. Does more training data help? That tells us if it's a bias problem or a variance problem.
Regularization. Sweep the C value in logistic regression and show how the score changes.
Calibration. Plot predicted probability against actual cover rate with seaborn. Are the predictions all bunched near 50%?
Coefficients. Show the sign and size of each one, and say whether they make football sense.
Step 5. Set up a real feature evaluation process

Teach me how pros decide if a feature is worth keeping, and build it as a reusable function in src/.

Test one feature at a time against the current model with the paired bootstrap.
Check stability across seasons.
Use permutation importance on the test folds.
Drop features that don't earn their place. Simpler is better here.
Step 6. Make a ranked list of what to try next

Rank candidate features and changes by how likely each one is to hold information the closing line misses, and how hard it is to build. Explain the reasoning for each. Consider these, but don't limit yourself to them.

Opponent-adjusted EPA, recency weighting, and success rate instead of raw EPA per play
Quarterback changes and starter injuries
Short weeks, travel distance, and time zone changes
Line movement from open to close
Changing the question. One option is to grade against the opening line instead of the close, since that's where pros find edges. Another is predicting margin and comparing it to the spread.
Rules for this session
Diagnosis comes before fixes. Stop after each step and wait for my go-ahead.
Run the skeptic agent at the end of Steps 3 and 6.
Write down the main lessons in concepts.md in plain words so I can explain them in an interview.