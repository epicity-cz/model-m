import pandas as pd


df = pd.read_csv(
    "https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/testy.csv")

df = df.iloc[30:]
df = df[["datum", "prirustkovy_pocet_testu"]]

POP = 10693939
HODONIN = 56103

df["prirustkovy_pocet_testu"] = df["prirustkovy_pocet_testu"] * HODONIN / POP

df = df.reset_index(drop=True)

print(df)

df.to_csv("tests.csv", index=False)
