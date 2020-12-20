import pandas as pd


def education(df):
    threshold_education = 0.002
    count_education_train = df.institution.value_counts(normalize=True)
    freq_education_train = count_education_train.drop('Школа').loc[count_education_train > threshold_education].mul(
        100).round(1).astype(str)
    groupped_education_train = df.loc[:, ['salary', 'institution']].groupby('institution').mean()
    groupped_education_train['normalized'] = (groupped_education_train - groupped_education_train.min()) / (
            groupped_education_train.max() - groupped_education_train.min())
    df['top_institution_by_freq'] = [
        groupped_education_train['normalized'].loc[x.institution] if x.institution in freq_education_train else 0 for
        _, x
        in df.iterrows()]
    salary_education_train = groupped_education_train['normalized'].sort_values(ascending=False).head(
        int(len(groupped_education_train['normalized']) * threshold_education))
    df['top_institution_by_salary'] = [
        groupped_education_train['normalized'].loc[x.institution] if x.institution in salary_education_train else 0 for
        _, x
        in df.iterrows()]
    return df


def employers(df):
    freq_employers = df.employer.value_counts()
    freq_employers = freq_employers.loc[freq_employers > 5].head(100)
    groupped_employer_freq = df.loc[:, ['salary_desired', 'employer']].groupby('employer').mean()
    groupped_employer_freq['normalized'] = (groupped_employer_freq - groupped_employer_freq.min()) / (
            groupped_employer_freq.max() - groupped_employer_freq.min())
    df['top_employer_by_freq'] = [
        groupped_employer_freq['normalized'].loc[x.employer] if x.employer in freq_employers else 0 for _, x
        in
        df.iterrows()]
    top_employers = groupped_employer_freq['normalized'].sort_values(ascending=False).head(100)
    df['top_employer_by_salary'] = [
        groupped_employer_freq['normalized'].loc[x.employer] if x.employer in top_employers else 0 for _, x in
        df.iterrows()]
    return df


df_train = pd.read_csv('train.csv', sep=';', dtype='unicode')
df_test = pd.read_csv('test.csv', sep=';', dtype='unicode')
train = df_train.loc[:, ['id', 'salary_desired', 'salary']]
train.iloc[:, 0] = [int(x) for x in train.iloc[:, 0]]
train.iloc[:, 1] = [int(x) for x in train.iloc[:, 1]]
train.iloc[:, 2] = [int(x) for x in train.iloc[:, 2]]
test = df_test.loc[:, ['id', 'salary_desired']]
test.iloc[:, 0] = [int(x) for x in test.iloc[:, 0]]
test.iloc[:, 1] = [int(x) for x in test.iloc[:, 1]]

df_education = pd.read_csv('education.csv', sep=';')
df_education.iloc[:, 0] = [int(x) for x in df_education.iloc[:, 0]]
merged_education_train = education(pd.merge(train, df_education, on='id', how='inner'))
merged_education_test = education(pd.merge(test, df_education, on='id', how='inner'))

df_lemmatized = pd.read_csv('df_lemmatized.csv').drop('Unnamed: 0', axis=1)
df_lemmatized_train = pd.merge(train, df_lemmatized, on='id', how='left').fillna('[]')
df_lemmatized_test = pd.merge(test, df_lemmatized, on='id', how='left').fillna('[]')

threshold_experience = 0.001

position_train = df_lemmatized_train.position.value_counts()
top_positions_train = position_train.head(int(len(position_train) * threshold_experience))
groupped_position_train = df_lemmatized_train.loc[:, ['salary', 'position']].groupby('position').mean()
groupped_position_train['normalized'] = (groupped_position_train - groupped_position_train.min()) / (
        groupped_position_train.max() - groupped_position_train.min())
df_lemmatized_train['mean_salary_positions'] = [
    groupped_position_train['normalized'].loc[x.position] if x.position in top_positions_train else 0 for _, x in
    df_lemmatized_train.iterrows()]

position_test = df_lemmatized_test.position.value_counts()
top_positions_test = position_test.head(int(len(position_test) * threshold_experience))
groupped_position_test = df_lemmatized_test.loc[:, ['salary_desired', 'position']].groupby('position').mean()
groupped_position_test['normalized'] = (groupped_position_test - groupped_position_test.min()) / (
        groupped_position_test.max() - groupped_position_test.min())
df_lemmatized_test['mean_salary_positions'] = [
    groupped_position_test['normalized'].loc[x.position] if x.position in top_positions_test else 0 for _, x in
    df_lemmatized_test.iterrows()]

skills_train = df_lemmatized_train.responsibilities.value_counts()
freq_positions_train = skills_train.head(int(len(position_train) * threshold_experience))
groupped_skills_train = df_lemmatized_train.loc[:, ['salary', 'responsibilities']].groupby('responsibilities').mean()
groupped_skills_train['normalized'] = (groupped_skills_train - groupped_skills_train.min()) / (
        groupped_skills_train.max() - groupped_skills_train.min())
df_lemmatized_train['mean_salary_skills'] = [
    groupped_skills_train['normalized'].loc[x.responsibilities] if x.responsibilities in freq_positions_train else 0 for
    _, x in df_lemmatized_train.iterrows()]

skills_test = df_lemmatized_test.responsibilities.value_counts()
freq_positions_test = skills_test.head(int(len(position_test) * threshold_experience))
groupped_skills_test = df_lemmatized_test.loc[:, ['salary_desired', 'responsibilities']].groupby(
    'responsibilities').mean()
groupped_skills_test['normalized'] = (groupped_skills_test - groupped_skills_test.min()) / (
        groupped_skills_test.max() - groupped_skills_test.min())
df_lemmatized_test['mean_salary_skills'] = [
    groupped_skills_test['normalized'].loc[x.responsibilities] if x.responsibilities in freq_positions_test else 0 for
    _, x in df_lemmatized_test.iterrows()]

df_employements = pd.read_csv('employements_mult.csv', sep=';').dropna()
df_employements.iloc[:, 0] = [int(x) for x in df_employements.iloc[:, 0]]

employers_train = pd.merge(train.loc[:, ['id', 'salary']], df_employements.loc[:, ['id', 'employer']], on='id',
                           how='left').fillna('')
employers_test = pd.merge(test.loc[:, ['id', 'salary_desired']], df_employements.loc[:, ['id', 'employer']], on='id',
                          how='left').fillna('')
employers_train = employers(employers_train)
employers_test = employers(employers_test)

main_df_train = pd.merge(merged_education_train,
                         df_lemmatized_train.loc[:, ['id', 'mean_salary_positions', 'mean_salary_skills']], on='id')
main_df_train = pd.merge(main_df_train,
                         employers_train.loc[:, ['id', 'top_employer_by_freq', 'top_employer_by_salary']], on='id')
main_df_test = pd.merge(merged_education_test,
                        df_lemmatized_test.loc[:, ['id', 'mean_salary_positions', 'mean_salary_skills']], on='id')
main_df_test = pd.merge(main_df_test, employers_test.loc[:, ['id', 'top_employer_by_freq', 'top_employer_by_salary']],
                        on='id')

main_df_train.to_csv('train_features.csv')
main_df_test.to_csv('test_features.csv')



