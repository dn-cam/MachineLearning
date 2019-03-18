import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

year1 = pd.read_csv('1year.csv')
year2 = pd.read_csv('2year.csv')
year3 = pd.read_csv('3year.csv')
year4 = pd.read_csv('4year.csv')
year5 = pd.read_csv('5year.csv')

year1.shape #(7027, 66)
year2.shape #(10173, 66)
year3.shape #(10503, 66)
year4.shape #(9792, 66)
year5.shape #(5910, 66)

#bankdata = pd.merge(da1)
year1 = pd.DataFrame(data = year1)
year2 = pd.DataFrame(data = year2)
year3 = pd.DataFrame(data = year3)
year4 = pd.DataFrame(data = year4)
year5 = pd.DataFrame(data = year5)

heatmap_data = year1.drop(['id', 'class'],axis = 1)
heatmap_data.head()

f, ax = plt.subplots(figsize=(30,30))
corr = heatmap_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='RdBu',square=True, ax=ax, annot = True)


bankrupt = year1.drop(['id'], axis = 1)
bankrupt.head()
#('Attr53', 'Attr5')

f, ax = plt.subplots(figsize=(30,30))
corr = bankrupt1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='RdBu',square=True, ax=ax, annot = True)

def convert_class_label_type_int(dfs):
    for i in range(5):
        colname = dfs[i].columns['class']
        col = getattr(dfs[i], colname)
        dfs[i]['class'] = col.astype(int)
        
convert_class_label_type_int(years)


years = [year1, year2, year3, year4, year5]
for i in range(len(years)):
    years[i] = years[i].replace('?', np.nan)
    
def convert_columns_type_float(dfs):
    for i in range(5):
        index = 1
        while(index<=64):
            colname = dfs[i].columns[index]
            col = getattr(dfs[i], colname)
            dfs[i][colname] = col.astype(float)
            index+=1
            
convert_columns_type_float(years)


from sklearn import preprocessing
years1_ = year1.drop(['id'], axis = 1)

for i in range(0,63):
    years1_.iloc[0:, i:i+1] = years1_.iloc[0:, 4:5].astype(float) 
    years1_.iloc[0:, i:i+1] = preprocessing.scale(years1_.iloc[0:, i:i+1])


    def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

top_500_correlations = get_top_abs_correlations(bankrupt1, 170)


top_500_correlations

print(top_500_correlations, file=f)

top_500_correlations_df = pd.DataFrame(top_500_correlations)

year1_0 = (year1['class'] == 1)
year_1_1= sum(year1_0 == True)
year_1_0= sum(year1_0 == False)

year2_0 = (year2['class'] == 1)
year_2_1= sum(year2_0 == True)
year_2_0= sum(year2_0 == False)

year3_0 = (year3['class'] == 1)
year_3_1= sum(year3_0 == True)
year_3_0= sum(year3_0 == False)

year4_0 = (year4['class'] == 1)
year_4_1= sum(year4_0 == True)
year_4_0= sum(year4_0 == False)

year5_0 = (year1['class'] == 1)
year_5_1= sum(year5_0 == True)
year_5_0= sum(year5_0 == False)


years = 5
zeroes = (year_1_0, year_2_0, year_3_0, year_4_0, year_5_0)
ones = (year_1_1, year_2_1, year_3_1, year_4_1, year_5_1)
ind = np.arange(years) 

width = 0.70

p1 = plt.bar(ind, zeroes, width)
p2 = plt.bar(ind, ones, width)

plt.ylabel('Proportions')
plt.title('Numbers of Bankrupted vs Non-bankrupted Companies')
plt.xticks(ind, ('Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'))
plt.yticks(np.arange(0, 12000, 1000))
plt.legend((p1[0], p2[0]), ('0', '1'))
plt.show()
r1_0 = year1[(year1['class'] == 1)].count()


missing_df_i = year2.columns[year2.isnull().any()].tolist()

msno.matrix(year1[missing_df_i], figsize=(20,5))

years = [year1, year2, year3, year4, year5]
for i in range(len(years)):
    years[i] = years[i].replace('?', np.nan)
    
    
import seaborn as sns

#f, ax = plt.subplots(figsize=(20,20))
year1 = pd.DataFrame(year1)
#sns.heatmap(year1.isnull(), ax = ax, cbar = False)
msno.matrix(years[0])
msno.matrix(years[1])
msno.matrix(years[2])
msno.matrix(years[3])
msno.matrix(years[4])



#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='RdBu',square=True, ax=ax, annot = True)


