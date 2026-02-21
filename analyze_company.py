import pandas as pd
from itertools import combinations

# import fastparquet

data = pd.read_parquet(path="veridion_entity_resolution_challenge.snappy.parquet", engine="fastparquet")  # or
# pyarrow, thought fastparquet is said to be faster

pd.set_option('display.max_columns', None)  # to get a grasp of what data is present in the columns

# Data profiling

# data.info()

# We can see that ~32k rows with company name, and we can divide the data into three groups of columns: The location,
# the social and the business groups Location columns (such as country, region, city, street, lat, lon) tells us
# where a company is located. This is important for our analysis since the same company can have multiple location
# around the world Social columns (such as email, phone, website, social media platforms) helps us in finding unique
# businesses. A company can have a single website, and some might be subsidiaries Business columns (such as business
# model, product type, sics/nace/isic codes) helps us in finding unique business but also if that company changed
# businesses We can also extract the fact that we are working with Active companies all over, and pretty much all of
# them have date of creation and update

data_describe = data.describe()
# print(data_describe)

# Describing the data gives us more insights: a important key that provides uniqueness is quite low: only 6.6k
# website domains. Unfortunately, social are quite low

# let's have a look of how a specific company name rows look like
fresh_burgers = data.loc[data["company_name"] == "Fresh Burger"]
# print(fresh_burgers)
# This looks scary: even if a row has the same city and same street name, it does not have the
# same postcode. Same post code does not have the same street. Only one record has street number, lat and long But
# there is one row that has sic_codes (entry 6026, sic_codes 5812) that basically tells us if it has a sic code,
# it might have a isic code/label and nace code/label. We should verify if that's true (a correlation between codes)


# let's find another business we can have some insights on

owens_liquors = data[data["company_name"].str.contains("Owens", na=False)]
# print(owens_liquors)
# this time I wanted to get a better understanding if there might be different names for the
# same company this is a case where the same company has the two locations (Myrtle Beach and Pawleys Island) because
# they have the same codes. The location in Hemingway does not tell us much, except the fact that has the same
# website_domain. For this case, I would leave this specific entry separated from the others, but is good to know
# that website domain can be considered an important feature in our analysis. We can conclude using this case that we
# can use both social and business codes to uniquely identify a business, and the address for multiple locations.

# let's take one more business to analyse separately.

indian_spices = data.loc[data["website_domain"] == "indianspices.com"]


# print(indian_spices)


def get_codes(df):
    codes = ['isic_v4_codes', 'sic_codes', 'nace_rev2_codes', 'naics_2022_primary_code']

    res = list(combinations(codes, 2))
    for element in res:
        percentage = len(df[df[element[0]].notna() & df[element[1]].notna()]) / len(df[df[element[0]].notna()]) * 100
        print(f"What the chance to have both {element[0]} and {element[1]} in the same row?: {percentage}")


# get_codes(data)
website_distribution = data["website_domain"].value_counts()
websites = website_distribution[website_distribution > 1]
# print(websites)


groups = data.groupby(
    ["website_domain", "company_name", "main_city", "main_street", "main_postcode"]).size().sort_values(ascending=False)
# print(groups)
pd.set_option('display.max_colwidth', None)
charles = data.loc[data["website_domain"] == "charlestonsailingadventures.com"]

locations = ['main_country', 'main_region', 'main_city_district', 'main_city',
             'main_postcode', 'main_street', 'main_street_number',
             'main_latitude', 'main_longitude']

codes = ['isic_v4_codes', 'sic_codes', 'nace_rev2_codes', 'naics_2022_primary_code']

businesses = ['main_business_category', 'main_industry', 'main_sector',
              'business_model', 'product_type']

socials = ['phone_numbers', 'primary_email', 'emails', 'website_domain',
           'facebook_url', 'linkedin_url', 'instagram_url', 'youtube_url']

names = ['company_name', 'company_legal_names', 'company_commercial_names', 'company_type']

time = ['created_at', 'last_updated_at']

all_cols = names + locations + codes + businesses + socials + time

df_clean = data[all_cols]
# print(df_clean.shape)
import re


def normalization(df):
    df_normalized = df.copy()
    company_suffixes = ["ltd.", "gmbh", "inc.", "inc", "corp.", "corp", "ltd", "llc", "llc."]
    social_url = ['facebook_url', 'linkedin_url', 'instagram_url', 'youtube_url']
    pattern1 = r'\s+(' + '|'.join(company_suffixes) + r')$'
    pattern2 = r'[^\d]+'

    for col in names:
        df_normalized[col] = df_normalized[col].str.lower().str.strip()
        df_normalized[col] = df_normalized[col].str.replace(pattern1, '', regex=True)

    df_normalized["main_postcode"] = df_normalized["main_postcode"].str.replace(pattern2, '', regex=True)
    df_normalized["main_latitude"] = df_normalized["main_latitude"].apply(
        lambda x: round(float(x), 2) if x is not None else x)
    df_normalized["main_longitude"] = df_normalized["main_longitude"].apply(
        lambda x: round(float(x), 2) if x is not None else x)

    for code in codes:
        df_normalized[code] = df_normalized[code].str.split("|").apply(
            lambda x: ", ".join(sorted([s.strip() for s in x])) if isinstance(x, list) else x)

    for social in social_url:
        df_normalized[social] = df_normalized[social].str.replace(r'https?://(www\.)?', '', regex=True).str.lower()

    return df_normalized


normalized = normalization(df_clean)

check = normalized.groupby("website_domain")["main_industry"].value_counts().sort_values(ascending=False)


# print(check)
# print(normalized)

# Tier one
def tier_one(df):
    df = df.copy()
    duplicates_tier1 = []
    group = df.groupby(["website_domain", "company_name"])
    for (website, name), group_df in group:
        uq = group_df.groupby(["main_city", "main_postcode", "main_region"]).ngroups
        if uq == 1:
            completeness = group_df.notna().sum(axis=1)
            master_row = completeness.idxmax()
            duplicates = group_df.index[group_df.index != master_row]
            duplicates_tier1.extend(duplicates)

    df.drop(duplicates_tier1, inplace=True)

    return df


df_1 = tier_one(normalized)
fresh = normalized.loc[normalized["company_name"] == "fresh burger"]
print(fresh.groupby(["main_city", "main_postcode", "main_region"]).ngroups)

#print(df_1.loc[df_1["company_name"] == "fresh burger"].shape)



# print(charles['locations'])
# this is a great case of data fragmentation. Different names (includes 'of' in some entries), codes for some
# entries, same website domain for each of them, but different URLs (I and i)


# We can conclude our specific analysis -> We first normalize the columns for company names.


# We apply multiple checks: if two rows have the same codes, we can pretty much say they are the same company. Then
# we have a look at the location. Third would be checks for business columns.
