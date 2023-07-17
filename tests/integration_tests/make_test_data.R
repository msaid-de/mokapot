set.seed(1)
N=1000
target_scores1 = rnorm(N, mean = -5, sd = 2)
target_scores2 = rnorm(N, mean = 0, sd = 3)
target_scores3 = rnorm(N, mean = 7, sd = 4)
decoy_scores1 = rnorm(N, mean = -9, sd = 2)
decoy_scores2 = rnorm(N, mean = 4, sd = 3)
decoy_scores3 = rnorm(N, mean = 12, sd = 4)
plot(density(target_scores1))
lines(density(decoy_scores1))
plot(density(target_scores2))
lines(density(decoy_scores2))
plot(density(target_scores3))
lines(density(decoy_scores3))

targets = data.frame(Label=1,
           score1=target_scores1,
           score2=target_scores2,
           score3=target_scores3)
decoys = data.frame(Label=-1,
           score1=decoy_scores1,
           score2=decoy_scores2,
           score3=decoy_scores3)
combined = rbind(targets, decoys)
NC = nrow(combined)
scans = 1:(NC/2)
combined$ScanNr = scans

peptides = sort(c(1:(NC/2), 1:(NC/2)))
combined$Peptide = paste(peptides)
combined$Proteins = "dummy"

combined = rbind(combined, combined)
combined$Specid = 1:nrow(combined)
combined = combined[,c(8, 1, 5,2:4,6:7)]
combined = combined[sample(nrow(combined)),]

write.table(combined, file = "test1.tab", sep = "\t", row.names = FALSE, quote = F)
combined$Specid = sample(combined$Specid)
write.table(combined, file = "test2.tab", sep = "\t", row.names = FALSE, quote = F)
