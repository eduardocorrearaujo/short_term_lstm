# short_term_lstm

| Parameter name | Type | Description |
|--|--|--|
| model | int | Model ID | 
| description | str or None | Prediction description |
| commit | str | Git commit hash to lastest version of Prediction's code in the Model's repository |
| predict_date | date _(YYYY-mm-dd)_ | Date when Prediction was generated |
| adm_0 | str _(ISO 3166-1)_ | Country code. Default: "BRA" |
| adm_1 | str _(UF)_ | State abbreviation. Example: "RJ" |
| adm_2 | int _(IBGE)_ | City geocode. Example: 3304557 |
| adm_3 | int _(IBGE)_ | - |
| prediction | dict _(JSON)_ | The Prediction result data. Example: [{"date": "2010-01-03","pred": 100,"lower_95": 65,"lower_90": 70,
"lower_80": 80,"lower_50": 90, "upper_50": 110,"upper_80": 120, "upper_90": 130,"upper_95": 135}] |
