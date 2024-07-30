# Categorizing Student Task Descriptions Using NLP

## Problem Overview
**Problem Statement:**
- Text data from ~700 students over a 15-week semester
- Students describe their research tasks up to 3 times per week
- Goal: Identify the most likely category assignment for each response

**Data Characteristics:**
- Up to 45 reports per student
- Descriptions are at least 10 words long
- 11 pre-defined categories
- Labeled dataset available for training

## Workflow Overview
**Workflow Steps:**

### Data Preparation:
1. Data Collection and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing

### Feature Engineering:
4. Text Embeddings

### Model Development:
5. Model Selection and Training
6. Model Evaluation

### Advanced Techniques:
7. Prompt Engineering
8. Large Language Models (LLMs)
9. Fine-Tuning LLMs

### Integration and Deployment:
10. Combining Approaches
11. Deployment and Monitoring

## 1- Data Collection and Preprocessing
**Steps:**
- Collect text data from student reports
- Combine and organize data into a suitable format (e.g., DataFrame)
- Ensure data quality by handling missing values and inconsistencies

**Tools:**
- Pandas for data manipulation
- Numpy for numerical operations

**Example:**
```python
import pandas as pd

# Sample data
data = {
    'student_id': [1, 1, 2, 2, 3],
    'week': [1, 2, 1, 2, 1],
    'report': [
        "Conducted a literature review on AI ethics",
        "Collected data from the field",
        "Analyzed the experimental results",
        "Wrote the introduction section of the paper",
        "Designed the experimental setup"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
```

## 2- Exploratory Data Analysis (EDA)
**Goals:**
- Understand the distribution of text data
- Visualize the frequency of categories
- Identify common keywords and phrases
- Find more optimal categories for new data

**Techniques:**
- Descriptive statistics
- Word clouds
- Frequency distribution plots

**Tools:**
- Matplotlib and Seaborn for visualization
- WordCloud library

**Examples:**

**Descriptive Statistics:**
```python
import pandas as pd

# Sample data
data = {'category': ["Writing", "Data Collection", "Data Analysis", "Writing", "Experimental Design"]}
df = pd.DataFrame(data)

# Descriptive statistics
category_counts = df['category'].value_counts()
print(category_counts)
```

**Word Cloud:**
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text
text = "Conducted a literature review on AI ethics Collected data from the field Analyzed the experimental results"

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

**Frequency Distribution Plot:**
```python
import matplotlib.pyplot as plt

# Sample data
data = {'category': ["Writing", "Data Collection", "Data Analysis", "Writing", "Experimental Design"]}
df = pd.DataFrame(data)

# Frequency distribution plot
category_counts = df['category'].value_counts()
category_counts.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Categories')
plt.show()
```
