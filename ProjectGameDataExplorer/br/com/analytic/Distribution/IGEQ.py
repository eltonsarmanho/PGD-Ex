'''
Created on 29 de mai de 2020

@author: elton
'''
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create bars
    barWidth = 1
    bars1 = [1.3, 4, 3.3,  2.1]
    bars2 = [3.2, 3.6, 3.5, 3.2]
    bars3 = [3.6, 2.8, 4,  2.5]
    bars4 = [3.6, 3.4, 3.4,  3.1]
    bars = bars1 + bars2 + bars3 + bars4
    N_dimensoes = int(len(bars)/4);
    # The X position of bars
    r1 = [1,6,11,16]
    r2 = [2,7,12,17]
    r3 = [3,8,13,18]
    r4 = [4,9,14,19]

    #'Challenge_4','Competence_4', 'Immersion_4','Flow_4', 'PositiveAffect_4','NegativeAffect_4','Calm_4','Happy_4','Anger_4','Sadness_4'

    r_result = r1 + r2 + r3 + r4
     
    # Create barplot
    plt.bar(r1, bars1, width = barWidth, color = (0.3,0.1,0.4,0.6), label='Session 1')
    plt.bar(r2, bars2, width = barWidth, color = (0.3,0.5,0.4,0.6), label='Session 2')
    plt.bar(r3, bars3, width = barWidth, color = (0.3,0.9,0.4,0.6), label='Session 3')
    plt.bar(r4, bars4, width = barWidth, color = (0.3,0.4,0.5,0.7), label='Session 4')
    
    # Note: the barplot could be created easily. See the barplot section for other examples.
     
    # Create legend
    plt.legend()
    print(N_dimensoes)
    # Text below each barplot with a rotation at 90°
    plt.xticks([r*5 + barWidth*2.5 for r in range(N_dimensoes)], ['Challenger','Competence', 'Immersion','Negative Affect'], rotation=90)
     
    # Create labels
    #label = ['n = 6', 'n = 25', 'n = 13', 'n = 36', 'n = 30', 'n = 11', 'n = 16', 'n = 37', 'n = 14', 'n = 4', 'n = 31', 'n = 34']
     
    # Text on the top of each barplot
    #for i in range(len(r4)):
    #    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)
     
    # Adjust the margins
    plt.subplots_adjust(bottom= 0.2, top = 0.98)
     
    # Show graphic
    plt.show()
