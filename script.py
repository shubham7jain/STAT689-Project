from main import runner
import matplotlib.pyplot as plt

classifiers_for_filtering_MF_and_CF = [
    'LogisticRegression',
    'RandomForestClassifier',
    'MLPClassifier'
]

classifiers_for_filtering_for_SF = [
    'LogisticRegression'
]

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
scores1 = []
scores2 = []
scores3 = []
scores4 = []
pE1_SF = []
pE1_MF = []
pE1_CF = []
pE2_SF = []
pE2_MF = []
pE2_CF = []

for i in range(len(noise_levels)):
    score1 = runner(classifiers_for_filtering_for_SF, noise_levels[i], False)
    scores1.append(score1)
    score2, pE1, pE2 = runner(classifiers_for_filtering_for_SF, noise_levels[i], True, 'SF')
    scores2.append(score2)
    pE1_SF.append(pE1)
    pE2_SF.append(pE2)
    score3, pE1, pE2 = runner(classifiers_for_filtering_MF_and_CF, noise_levels[i], True, 'MF')
    scores3.append(score3)
    pE1_MF.append(pE1)
    pE2_MF.append(pE2)
    score4, pE1, pE2 = runner(classifiers_for_filtering_MF_and_CF, noise_levels[i], True, 'CF')
    scores4.append(score4)
    pE1_CF.append(pE1)
    pE2_CF.append(pE2)

for i in range(len(noise_levels)):
    print('Noise Level : ', noise_levels[i])
    print('Filter: None : Accuracy = ', scores1[i])
    print('Filter: SF : Accuracy = ', scores2[i], ', P(E1) = ', pE1_SF[i], ', P(E2) = ', pE2_SF[i])
    print('Filter: MF : Accuracy = ', scores3[i], ', P(E1) = ', pE1_MF[i], ', P(E2) = ', pE2_MF[i])
    print('Filter: CF : Accuracy = ', scores4[i], ', P(E1) = ', pE1_CF[i], ', P(E2) = ', pE2_CF[i])

plt.plot(noise_levels, scores1, 'r--', label='None')
plt.plot(noise_levels, scores2, 'b--', label='SF')
plt.plot(noise_levels, scores3, 'g--', label='MF')
plt.plot(noise_levels, scores4, 'y--', label='CF')
plt.title("Accuracy of the MNIST dataset")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.legend(loc='upper right')
plt.show()