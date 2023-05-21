import barnet_unemployement
import sunlight_to_sql
import housing_combined
import imd2SQL
import unemployement_to_sql


def main():
    print("Unemployement by LSOA by month in barnet")
    barnet_unemployement.main()
    print("sunlight")
    sunlight_to_sql.main()
    print("housing_combined")
    housing_combined.main()
    print("IMD")
    imd2SQL.main()
    print("Unemployement By Ward")
    unemployement_to_sql.main()

if __name__ == "__main__":
    main()
