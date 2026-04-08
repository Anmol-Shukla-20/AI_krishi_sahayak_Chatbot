# AI_krishi_sahayak_Chatbot
This is a project based on Crop recommendation based on machine learning(Random Forest) and cleaner UI.It is also powered by chatbot features to answer queries.

```python
import numpy as np
import pandas as pd
```

### 1.Loading/Importing Data


```python
crop = pd.read_csv(r"C:\Users\HP\Desktop\DETD_Vppr_Sir\Notebook_analysis_crop_recommendation\Crop_recommendation.csv")
crop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
      <td>rice</td>
    </tr>
  </tbody>
</table>
</div>




```python
crop.tail()  # to see the last values of the dataset.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2195</th>
      <td>107</td>
      <td>34</td>
      <td>32</td>
      <td>26.774637</td>
      <td>66.413269</td>
      <td>6.780064</td>
      <td>177.774507</td>
      <td>coffee</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>99</td>
      <td>15</td>
      <td>27</td>
      <td>27.417112</td>
      <td>56.636362</td>
      <td>6.086922</td>
      <td>127.924610</td>
      <td>coffee</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>118</td>
      <td>33</td>
      <td>30</td>
      <td>24.131797</td>
      <td>67.225123</td>
      <td>6.362608</td>
      <td>173.322839</td>
      <td>coffee</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>117</td>
      <td>32</td>
      <td>34</td>
      <td>26.272418</td>
      <td>52.127394</td>
      <td>6.758793</td>
      <td>127.175293</td>
      <td>coffee</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>104</td>
      <td>18</td>
      <td>30</td>
      <td>23.603016</td>
      <td>60.396475</td>
      <td>6.779833</td>
      <td>140.937041</td>
      <td>coffee</td>
    </tr>
  </tbody>
</table>
</div>




```python
crop.shape
```




    (2200, 8)




```python
crop.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2200 entries, 0 to 2199
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   N            2200 non-null   int64  
     1   P            2200 non-null   int64  
     2   K            2200 non-null   int64  
     3   temperature  2200 non-null   float64
     4   humidity     2200 non-null   float64
     5   ph           2200 non-null   float64
     6   rainfall     2200 non-null   float64
     7   label        2200 non-null   object 
    dtypes: float64(4), int64(3), object(1)
    memory usage: 137.6+ KB
    

### Checking for Null Values


```python
crop.isnull().sum()
```




    N              0
    P              0
    K              0
    temperature    0
    humidity       0
    ph             0
    rainfall       0
    label          0
    dtype: int64



### Checking For Duplicates in the dataset(if Present)


```python
crop.duplicated().sum()
```




    np.int64(0)




```python
crop.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2200.000000</td>
      <td>2200.000000</td>
      <td>2200.000000</td>
      <td>2200.000000</td>
      <td>2200.000000</td>
      <td>2200.000000</td>
      <td>2200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.551818</td>
      <td>53.362727</td>
      <td>48.149091</td>
      <td>25.616244</td>
      <td>71.481779</td>
      <td>6.469480</td>
      <td>103.463655</td>
    </tr>
    <tr>
      <th>std</th>
      <td>36.917334</td>
      <td>32.985883</td>
      <td>50.647931</td>
      <td>5.063749</td>
      <td>22.263812</td>
      <td>0.773938</td>
      <td>54.958389</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>8.825675</td>
      <td>14.258040</td>
      <td>3.504752</td>
      <td>20.211267</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>28.000000</td>
      <td>20.000000</td>
      <td>22.769375</td>
      <td>60.261953</td>
      <td>5.971693</td>
      <td>64.551686</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>51.000000</td>
      <td>32.000000</td>
      <td>25.598693</td>
      <td>80.473146</td>
      <td>6.425045</td>
      <td>94.867624</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>84.250000</td>
      <td>68.000000</td>
      <td>49.000000</td>
      <td>28.561654</td>
      <td>89.948771</td>
      <td>6.923643</td>
      <td>124.267508</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.000000</td>
      <td>145.000000</td>
      <td>205.000000</td>
      <td>43.675493</td>
      <td>99.981876</td>
      <td>9.935091</td>
      <td>298.560117</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Exploratory Data Analysis(EDA)


```python
numeric_crop = crop.select_dtypes(include={'float64','int64'}) # Selecting only Numeric Columns
corr = numeric_crop.corr()      
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N</th>
      <td>1.000000</td>
      <td>-0.231460</td>
      <td>-0.140512</td>
      <td>0.026504</td>
      <td>0.190688</td>
      <td>0.096683</td>
      <td>0.059020</td>
    </tr>
    <tr>
      <th>P</th>
      <td>-0.231460</td>
      <td>1.000000</td>
      <td>0.736232</td>
      <td>-0.127541</td>
      <td>-0.118734</td>
      <td>-0.138019</td>
      <td>-0.063839</td>
    </tr>
    <tr>
      <th>K</th>
      <td>-0.140512</td>
      <td>0.736232</td>
      <td>1.000000</td>
      <td>-0.160387</td>
      <td>0.190859</td>
      <td>-0.169503</td>
      <td>-0.053461</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>0.026504</td>
      <td>-0.127541</td>
      <td>-0.160387</td>
      <td>1.000000</td>
      <td>0.205320</td>
      <td>-0.017795</td>
      <td>-0.030084</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>0.190688</td>
      <td>-0.118734</td>
      <td>0.190859</td>
      <td>0.205320</td>
      <td>1.000000</td>
      <td>-0.008483</td>
      <td>0.094423</td>
    </tr>
    <tr>
      <th>ph</th>
      <td>0.096683</td>
      <td>-0.138019</td>
      <td>-0.169503</td>
      <td>-0.017795</td>
      <td>-0.008483</td>
      <td>1.000000</td>
      <td>-0.109069</td>
    </tr>
    <tr>
      <th>rainfall</th>
      <td>0.059020</td>
      <td>-0.063839</td>
      <td>-0.053461</td>
      <td>-0.030084</td>
      <td>0.094423</td>
      <td>-0.109069</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pip install seaborn 
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot= True,cbar=True,cmap = "coolwarm")
plt.title('Feature Correlation Heatmap')
plt.show()
```


    
asset/image_1.png
    



```python
crop['label'].value_counts()
```




    label
    rice           100
    maize          100
    chickpea       100
    kidneybeans    100
    pigeonpeas     100
    mothbeans      100
    mungbean       100
    blackgram      100
    lentil         100
    pomegranate    100
    banana         100
    mango          100
    grapes         100
    watermelon     100
    muskmelon      100
    apple          100
    orange         100
    papaya         100
    coconut        100
    cotton         100
    jute           100
    coffee         100
    Name: count, dtype: int64




```python
sns.distplot(crop['N'])  # to chek the spread out of the data.
plt.show()
```

    C:\Users\HP\AppData\Local\Temp\ipykernel_11280\1908739173.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(crop['N'])  # to chek the spread out of the data.
    


    
asset/image_2.png
    



```python
sns.histplot(crop['N'])
plt.show()
```


    
asset/image_3.png


## 3. Encoding Labels into Numerical values


```python
crop_dict ={
    'rice'       :   1,
    'maize'      :   2,
    'chickpea'   :   3,
    'kidneybeans':   4,
    'pigeonpeas' :   5,
    'mothbeans'  :   6,
    'mungbean'   :   7,
    'blackgram'  :   8,
    'lentil'     :   9,
    'pomegranate':   10,
    'banana'     :   11,
    'mango'      :   12,
    'grapes'     :   13,
    'watermelon' :   14,
    'muskmelon'  :   15,
    'apple'      :   16,
    'orange'     :   17,
    'papaya'     :   18,
    'coconut'    :   19,
    'cotton'     :   20,
    'jute'       :   21,
    'coffee'     :   22,
}

crop['crop_num'] = crop['label'].map(crop_dict)
```


```python
reverse_crop_dict = {v: k for k, v in crop_dict.items()}  # return the items stored in key format..
```


```python
# crop['crop_num'].value_counts()  --> this prints the count of crop in the dataset..
# same is shown below also...

target_col = 'label'
crop[target_col].value_counts().plot(kind='bar', figsize=(12,5), title='Crop Distribution')
plt.show()
```


    
asset/image_4.png
    



```python
# crop.drop(['label'],axis = 1,inplace = True)
crop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
      <th>label</th>
      <th>crop_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
      <td>rice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
      <td>rice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
      <td>rice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
      <td>rice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
      <td>rice</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Splitting the Data into training and Testing.


```python
# Input features.
X = crop.drop(['crop_num','label'],axis = 1) 
# drops the crop num,label assigned to each crop and this would be given as an output variable.

# Output features.
y =  crop['crop_num']
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2195</th>
      <td>107</td>
      <td>34</td>
      <td>32</td>
      <td>26.774637</td>
      <td>66.413269</td>
      <td>6.780064</td>
      <td>177.774507</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>99</td>
      <td>15</td>
      <td>27</td>
      <td>27.417112</td>
      <td>56.636362</td>
      <td>6.086922</td>
      <td>127.924610</td>
    </tr>
    <tr>
      <th>2197</th>
      <td>118</td>
      <td>33</td>
      <td>30</td>
      <td>24.131797</td>
      <td>67.225123</td>
      <td>6.362608</td>
      <td>173.322839</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>117</td>
      <td>32</td>
      <td>34</td>
      <td>26.272418</td>
      <td>52.127394</td>
      <td>6.758793</td>
      <td>127.175293</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>104</td>
      <td>18</td>
      <td>30</td>
      <td>23.603016</td>
      <td>60.396475</td>
      <td>6.779833</td>
      <td>140.937041</td>
    </tr>
  </tbody>
</table>
<p>2200 rows × 7 columns</p>
</div>




```python
y.shape
```




    (2200,)




```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)
```


```python
X_train.shape
```




    (1540, 7)




```python
X_test.shape
```




    (660, 7)



### Training Dataset👇


```python
X_train   # shows the training dataset 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1102</th>
      <td>21</td>
      <td>26</td>
      <td>27</td>
      <td>27.003155</td>
      <td>47.675254</td>
      <td>5.699587</td>
      <td>95.851183</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>29</td>
      <td>35</td>
      <td>28</td>
      <td>28.347161</td>
      <td>53.539031</td>
      <td>6.967418</td>
      <td>90.402604</td>
    </tr>
    <tr>
      <th>141</th>
      <td>60</td>
      <td>44</td>
      <td>23</td>
      <td>24.794708</td>
      <td>70.045567</td>
      <td>5.722580</td>
      <td>76.728601</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>80</td>
      <td>77</td>
      <td>49</td>
      <td>26.054330</td>
      <td>79.396545</td>
      <td>5.519088</td>
      <td>113.229737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1638</th>
      <td>10</td>
      <td>5</td>
      <td>5</td>
      <td>21.213070</td>
      <td>91.353492</td>
      <td>7.817846</td>
      <td>112.983436</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>108</td>
      <td>94</td>
      <td>47</td>
      <td>27.359116</td>
      <td>84.546250</td>
      <td>6.387431</td>
      <td>90.812505</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>11</td>
      <td>36</td>
      <td>31</td>
      <td>27.920633</td>
      <td>51.779659</td>
      <td>6.475449</td>
      <td>100.258567</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>11</td>
      <td>124</td>
      <td>204</td>
      <td>13.429886</td>
      <td>80.066340</td>
      <td>6.361141</td>
      <td>71.400430</td>
    </tr>
    <tr>
      <th>860</th>
      <td>32</td>
      <td>78</td>
      <td>22</td>
      <td>23.970814</td>
      <td>62.355576</td>
      <td>7.007038</td>
      <td>53.409060</td>
    </tr>
  </tbody>
</table>
<p>1540 rows × 7 columns</p>
</div>



### Testing Dataset👇


```python
X_test   # shows testing data..
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1451</th>
      <td>101</td>
      <td>17</td>
      <td>47</td>
      <td>29.494014</td>
      <td>94.729813</td>
      <td>6.185053</td>
      <td>26.308209</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>98</td>
      <td>8</td>
      <td>51</td>
      <td>26.179346</td>
      <td>86.522581</td>
      <td>6.259336</td>
      <td>49.430510</td>
    </tr>
    <tr>
      <th>1761</th>
      <td>59</td>
      <td>62</td>
      <td>49</td>
      <td>43.360515</td>
      <td>93.351916</td>
      <td>6.941497</td>
      <td>114.778071</td>
    </tr>
    <tr>
      <th>1735</th>
      <td>44</td>
      <td>60</td>
      <td>55</td>
      <td>34.280461</td>
      <td>90.555616</td>
      <td>6.825371</td>
      <td>98.540477</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>30</td>
      <td>137</td>
      <td>200</td>
      <td>22.914300</td>
      <td>90.704756</td>
      <td>5.603413</td>
      <td>118.604465</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>398</th>
      <td>27</td>
      <td>63</td>
      <td>19</td>
      <td>20.934099</td>
      <td>21.189301</td>
      <td>5.562202</td>
      <td>133.191442</td>
    </tr>
    <tr>
      <th>584</th>
      <td>20</td>
      <td>50</td>
      <td>22</td>
      <td>30.996947</td>
      <td>46.426937</td>
      <td>9.406888</td>
      <td>38.315979</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>45</td>
      <td>47</td>
      <td>55</td>
      <td>38.419163</td>
      <td>91.142204</td>
      <td>6.751453</td>
      <td>119.265388</td>
    </tr>
    <tr>
      <th>292</th>
      <td>39</td>
      <td>76</td>
      <td>76</td>
      <td>19.968375</td>
      <td>15.573244</td>
      <td>8.135901</td>
      <td>69.157591</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>103</td>
      <td>16</td>
      <td>49</td>
      <td>24.067315</td>
      <td>81.640753</td>
      <td>6.915717</td>
      <td>51.752124</td>
    </tr>
  </tbody>
</table>
<p>660 rows × 7 columns</p>
</div>




```python
y_train   # shows train data output variables
```




    1102    12
    1159    12
    141      2
    1004    11
    2        1
            ..
    1638    17
    1095    11
    1130    12
    1294    13
    860      9
    Name: crop_num, Length: 1540, dtype: int64



### New Change...


```python
le = LabelEncoder()
y_enc = le.fit_transform(y)
```

### Standardisation


```python
# more better as it considers the group of values altogether..
```


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# sc.fit(X_train)

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```


```python
X_train_sc
```




    array([[-8.14149162e-01, -8.22608476e-01, -4.17586751e-01, ...,
            -1.10914730e+00, -1.00850068e+00, -1.14762954e-01],
           [-5.99794073e-01, -5.52511028e-01, -3.98018725e-01, ...,
            -8.39738838e-01,  6.40463882e-01, -2.12947619e-01],
           [ 2.30831896e-01, -2.82413580e-01, -4.95858854e-01, ...,
            -8.13537964e-02, -9.78595756e-01, -4.59356367e-01],
           ...,
           [-1.08209302e+00, -5.22500201e-01, -3.39314648e-01, ...,
            -9.20572349e-01,  6.00471872e-04, -3.53408620e-02],
           [-1.08209302e+00,  2.11845263e+00,  3.04595380e+00, ...,
             3.79045864e-01, -1.48070939e-01, -5.55371242e-01],
           [-5.19410914e-01,  7.37954558e-01, -5.15426879e-01, ...,
            -4.34666852e-01,  6.91994073e-01, -8.79579938e-01]],
          shape=(1540, 7))



## Training Models


```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report

```


```python
# Creating Instances for all the models

models = {
    'Logistic Regression': LogisticRegression(),
    # 'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

```


```python
results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f'{name} Accuracy: {acc:.4f}')
```

    Logistic Regression Accuracy: 0.9591
    Support Vector Machine Accuracy: 0.9727
    K-Nearest Neighbors Accuracy: 0.9682
    Decision Tree Accuracy: 0.9848
    Random Forest Accuracy: 0.9924
    Bagging Accuracy: 0.9909
    AdaBoost Accuracy: 0.1939
    Gradient Boosting Accuracy: 0.9833
    Extra Trees Accuracy: 0.9258
    


```python
# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print('Best model:', best_model_name, 'with accuracy:', results[best_model_name])

# Evaluate best model
preds = best_model.predict(X_test_sc)
print(classification_report(y_test, preds, zero_division=0))
```

    Best model: Random Forest with accuracy: 0.9924242424242424
                  precision    recall  f1-score   support
    
               1       1.00      0.82      0.90        28
               2       1.00      1.00      1.00        26
               3       1.00      1.00      1.00        34
               4       1.00      1.00      1.00        36
               5       1.00      1.00      1.00        37
               6       1.00      1.00      1.00        34
               7       1.00      1.00      1.00        30
               8       1.00      1.00      1.00        26
               9       1.00      1.00      1.00        22
              10       1.00      1.00      1.00        38
              11       1.00      1.00      1.00        26
              12       1.00      1.00      1.00        32
              13       1.00      1.00      1.00        23
              14       1.00      1.00      1.00        23
              15       1.00      1.00      1.00        24
              16       1.00      1.00      1.00        34
              17       1.00      1.00      1.00        25
              18       1.00      1.00      1.00        37
              19       1.00      1.00      1.00        33
              20       1.00      1.00      1.00        28
              21       0.87      1.00      0.93        34
              22       1.00      1.00      1.00        30
    
        accuracy                           0.99       660
       macro avg       0.99      0.99      0.99       660
    weighted avg       0.99      0.99      0.99       660
    
    


```python
# Save model
pipeline_obj = {
    'scaler': sc,
    'label_encoder': le,
    'model': best_model
}
```


```python
# print(best_model)
# print(sc)
# print(le)
```


```python
# Saving model and scalers
import pickle
pickle.dump(pipeline_obj, open("model.pkl", "wb"))
```


```python
# Saving model and scalers
import pickle
pickle.dump(best_model, open("crop_model.pkl", "wb"))
pickle.dump(sc, open("standscaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

```

## Developing Prediction System


```python
# Prediction example
example = {
    'N': 28,
    'P': 59,
    'K': 22,
    'temperature': 30,
    'humidity': 52.799,
    'ph': 7.05,
    'rainfall': 171
}
import pandas as pd
s = pd.DataFrame([example])
s_scaled = pipeline_obj['scaler'].transform(s)
pred = pipeline_obj['model'].predict(s_scaled)

pred_crop= reverse_crop_dict[int(pred[0])]
print('Predicted crop:', pred_crop)
```

    Predicted crop: pigeonpeas
    


```python
print(pipeline_obj['label_encoder'].classes_)

```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
    


```python
example = {
    'N': 65,
    'P': 51,
    'K': 32,
    'temperature': 40,
    'humidity': 80.4,
    'ph': 6.4,
    'rainfall': 95
}
import pandas as pd
s = pd.DataFrame([example])
s_scaled = pipeline_obj['scaler'].transform(s)
pred = pipeline_obj['model'].predict(s_scaled)

pred_crop= reverse_crop_dict[int(pred[0])]
print('Predicted crop:', pred_crop)
```

    Predicted crop: jute
    


```python

```


```python

```

