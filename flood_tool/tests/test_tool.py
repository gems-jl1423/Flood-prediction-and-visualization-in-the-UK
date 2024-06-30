"""Test flood tool."""

import numpy as np

from pytest import mark

import flood_tool.tool as tool
import pandas as pd
import io

testtool = tool.Tool()


def test_lookup_easting_northing():
    """Check"""

    data = testtool.lookup_easting_northing(["M34 7QL"])

    assert np.isclose(data.iloc[0].easting, 393470).all()
    assert np.isclose(data.iloc[0].northing, 394371).all()


@mark.xfail  # We expect this test to fail until we write some code for it.
def test_lookup_lat_long():
    """Check"""

    data = testtool.lookup_lat_long(["M34 7QL"])

    assert np.isclose(data.iloc[0].latitude, 53.4461, rtol=1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -2.0997, rtol=1.0e-3).all()


def test_lookup_lat_long():
    lat_data = testtool.lookup_lat_long(["M34 7QL"])
    assert np.isclose(lat_data.iloc[0].latitude, 53.446064)
    assert np.isclose(lat_data.iloc[0].longitude, -2.099783)


def test_predict_flood_class_from_postcode():
    postcodes = ["M34 7QL"]
    assert type(postcodes) == list 
    flood_predictions = testtool.predict_flood_class_from_postcode(postcodes)
    assert flood_predictions.index[0] == "M34 7QL"
    
def test_predict_historic_flooding():
    postcodes = ["M34 7QL"]
    assert type(postcodes) == list
    flood_predictions = testtool.predict_historic_flooding(postcodes)
    assert flood_predictions.index[0] == "M34 7QL"
    
    #assert istype.int 
    
def test_predict_local_authority():
    auth_predictions = testtool.predict_local_authority()
    #assert *output is of right type
    #assert that the output eastings == the _postcoded eastings 
    #assert value is correct 
    
    
def test_predict_total_value():
    val_predictions = testtool.predict_total_value()
    #assert val_predictions['postcode'] 

def test_predict_annual_flood_risk():
    pass

def test_predict_median_house_price():
    postcodes = ['M34 7QL', 'OL9 7NA']
    method = "house_price_rf"
    price_predictions = testtool.predict_median_house_price(postcodes=postcodes, method=method)

    assert isinstance(price_predictions, pd.Series)
    assert price_predictions.index.tolist() == postcodes
    assert all(isinstance(value, (int, float)) for value in price_predictions.values)

def test_add_population_to_df():
    data = {
        "postcode": ["OL9 7NS", "WV13 2LR", "LS12 1LZ", "SK15 1TS", "TS17 9NN"],
        "easting": [390978, 396607, 427859, 395560, 445771],
        "northing": [403269, 298083, 432937, 397900, 515362],
        "soilType": ["Unsurveyed/Urban"] * 5,
        "elevation": [130, 130, 60, 120, 20],
        "localAuthority": ["Oldham", "Walsall", "Leeds", "Tameside", "Stockton-on-Tees"],
        "riskLabel": [1, 1, 1, 1, 1],
        "medianPrice": [119100.0, 84200.0, 134900.0, 170200.0, 190600.0],
        "historicallyFlooded": [False] * 5,
    }
    df = pd.DataFrame(data)
    df = testtool.add_population_to_df(df)

    assert 'postcodeSector' in df.columns
    assert 'average_pop' in df.columns
    assert (df['postcode'].str[:-2] == df['postcodeSector']).all()


def test_add_pets_to_df():
    data = {
        "postcode": ["OL9 7NS", "WV13 2LR", "LS12 1LZ", "SK15 1TS", "TS17 9NN"],
        "easting": [390978, 396607, 427859, 395560, 445771],
        "northing": [403269, 298083, 432937, 397900, 515362],
        "soilType": ["Unsurveyed/Urban"] * 5,
        "elevation": [130, 130, 60, 120, 20],
        "localAuthority": ["Oldham", "Walsall", "Leeds", "Tameside", "Stockton-on-Tees"],
        "riskLabel": [1, 1, 1, 1, 1],
        "medianPrice": [119100.0, 84200.0, 134900.0, 170200.0, 190600.0],
        "historicallyFlooded": [False] * 5,
    }
    df = pd.DataFrame(data)
    df = testtool.add_pets_to_df(df)

    assert 'postcodeDistrict' in df.columns
    assert 'avg_pets_per_household' in df.columns
    assert (df['postcode'].str.split().str[0] == df['postcodeDistrict']).all()


def test_find_closest_region_feas():
    data = {
        "postcode": ["OL9 7NS", "WV13 2LR", "LS12 1LZ", "SK15 1TS", "TS17 9NN"],
        "easting": [390978, 396607, 427859, 395560, 445771],
        "northing": [403269, 298083, 432937, 397900, 515362],
        "soilType": ["Unsurveyed/Urban"] * 5,
        "elevation": [130, 130, 60, 120, 20],
        "localAuthority": ["Oldham", "Walsall", "Leeds", "Tameside", "Stockton-on-Tees"],
        "riskLabel": [1, 1, 1, 1, 1],
        "medianPrice": [119100.0, 84200.0, 134900.0, 170200.0, 190600.0],
        "historicallyFlooded": [False] * 5,
    }
    dataset = pd.DataFrame(data)
    dataset = testtool.add_population_to_df(dataset)
    dataset = testtool.add_pets_to_df(dataset)

    dataset.drop(columns=['soilType', 'elevation', 'localAuthority', 'riskLabel', 'medianPrice',
                          'historicallyFlooded', 'postcodeSector', 'postcodeDistrict'], inplace=True)
    postcodes = ['OL9 7NA', 'SK15 1TS', 'WV13 2LR', 'SW1A 1AA']
    postcodes_in_dataset = dataset[dataset['postcode'].isin(postcodes)]

    df = pd.DataFrame({'postcode': postcodes})
    df = pd.merge(df, postcodes_in_dataset, on='postcode', how='left')
    features = df[['postcode', 'easting', 'northing', 'average_pop', 'avg_pets_per_household']]
    features = testtool.find_closest_region_feas(dataset, features)

    assert features.shape[0] == len(postcodes)
    assert features.notna().all().all()


# Convenience function to run tests directly.
if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
    test_lookup_lat_long()
    test_predict_flood_class_from_postcode()
    test_add_population_to_df()
    test_find_closest_region_feas()
    test_predict_median_house_price()
