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

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
scores1 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
scores2 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
scores3 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
scores4 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# for i in range(len(noise_levels)):
#     scores1.append(runner(classifiers_for_filtering_for_SF, noise_levels[i], False))
#     scores2.append(runner(classifiers_for_filtering_for_SF, noise_levels[i], True, 'SF'))
#     scores3.append(runner(classifiers_for_filtering_MF_and_CF, noise_levels[i], True, 'MF'))
#     scores4.append(runner(classifiers_for_filtering_MF_and_CF, noise_levels[i], True, 'CF'))
#
# for i in range(len(noise_levels)):
#     print('Noise Level : ', noise_levels[i])
#     print('None : ', scores1[i])
#     print('SF : ', scores2[i])
#     print('MF : ', scores3[i])
#     print('CF : ', scores4[i])

plt.plot(noise_levels, scores1, 'r--', label='None')
plt.plot(noise_levels, scores2, 'b--', label='SF')
plt.plot(noise_levels, scores3, 'g--', label='MF')
plt.plot(noise_levels, scores4, 'y--', label='CF')
plt.title("Accuracy of the MNIST dataset")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.legend(loc='upper right')
plt.show()