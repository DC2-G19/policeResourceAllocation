import pandas as pd
import burglaries_per_month_by_lsoa
import numpy as np
from tqdm import tqdm


# def main():
#     shiftDct = {}
#     data = burglaries_per_month_by_lsoa.main()
#     for key in tqdm(data.keys()):
#         burglaries, unemployement = data[key]
#         burglaries = burglaries.reset_index()
#         burglaries = burglaries.rename(column={0:"burglaries_per_month"})
#         burglaries = burglaries.set_index("date")
#         burglaries.join(unemployement, how='inner')
#         burglaries.sort_index(inplace=True)
#         burglaries["pct_value"] = burglaries["value"].pct_change()
#         burglaries["pct_l1"] = burglaries['pct_value'].shift(1)
#         burglaries["pct_l2"] = burglaries['pct_value'].shift(2)
#         burglaries["pct_l3"] = burglaries['pct_value'].shift(3)
#         burglaries["pct_l4"]= burglaries['pct_value'].shift(4)
#         burglaries["pct_l5"] = burglaries['pct_value'].shift(5)
#         burglaries["pct_l6"] = burglaries['pct_value'].shift(6)
#         burglaries["pct_l7"] = burglaries['pct_value'].shift(7)
#         burglaries = burglaries[burglaries.index < "2020"]
#         burglaries = burglaries.fillna(0)
#         burglaries.replace([np.inf, -np.inf], 0, inplace=True)
#         shiftDct[key] = burglaries
#     return shiftDct

def main():
    shiftDct = {}
    data = burglaries_per_month_by_lsoa.main()
    
    for key, (burglaries, unemployement) in tqdm(data.items()):
        burglaries = burglaries.reset_index().rename(columns={0: "burglaries_per_month"}).set_index("date")
        burglaries = burglaries.join(unemployement, how='inner').sort_index()
        burglaries = burglaries[burglaries.index < "2020"].fillna(0).replace([np.inf, -np.inf], 0)
        
        for i in range(1, 8):
            burglaries[f"pct_l{i}"] = burglaries['value'].pct_change(periods=i)
        
        shiftDct[key] = burglaries
        
    return shiftDct
if __name__ == "__main__":
    print(main())