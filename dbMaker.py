import barnet_unemployement
import sunlight_to_sql
import housing_combined
import imd2SQL
import unemployement_to_sql
import crimeReplacer

def main():
    print("Unemployement by LSOA by Month in Barnet")
    barnet_unemployement.main()
    print("Sunlight")
    sunlight_to_sql.main()
    print("Housing Combined")
    housing_combined.main()
    print("IMD")
    imd2SQL.main()
    print("Unemployement by Ward")
    unemployement_to_sql.main()
    print("Replace Main Table")
    crimeReplacer.main()


if __name__ == "__main__":
    main()
