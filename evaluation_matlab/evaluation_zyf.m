feat1_t = features(4, :)';
feat2_t = features(5, :)';
scores_t = distance.compute_cosine_score(feat1_t, feat2_t)