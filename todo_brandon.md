s3z... 50k dmawm, 76.9 +- 13.7 sd.  our current results ~59 as baseline.
- [ ] backprop through critic to model
- [ ] joint prediction backprop to single agent.
- [ ] try experiments with lower/higher sigreg parameter weighting values. compare performance for hyperparameter tuning after 50k steps with dmawm results of 76.9 +-13.7 and our current results ~59 as baseline.
- [ ] Look into dmawm and see if anything special with interaction across agents? Try running with old or modified interaction module we stopped using temporarily for better results.
- [ ] Generally look into dmawm and see if there is anything they have we are missing. Also, check for any ``cheats'' in their code that may overinflate their results. Hopefully, these are not present and probably not, but worthwhile checking.
- [ ] Explore paired counterfactual action-effect supervision: branch factual and alternative legal actions from the same simulator/controller state with shared exogenous randomness, supervise each prediction with its own successor target or the observed target difference, and allow outcome-equivalent actions to have equal predictions.
