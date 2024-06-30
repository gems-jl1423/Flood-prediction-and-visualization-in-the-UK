
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .geo import *  # noqa: F401, F403

__all__ = [
    "Tool",
    "_data_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")


# dictionaries with keys of short name and values of long name fof
# classification/regression methods

# You should add your own methods here
flood_class_from_postcode_methods = {
    "better_selection": "Flood Class KNN & Random Forest Classifier",
}
flood_class_from_location_methods = {
    "better_selection": "Flood Class KNN & Random Forest Classifier",
}
historic_flooding_methods = {
    "historic_rf": "Historic Random Forest Classifier",
}
house_price_methods = {
    "house_price_rf": "House Price Random Forest Regressor",
}
local_authority_methods = {
    "local_rf": "Local Authority Random Forest Classifier",
}



class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, unlabelled_unit_data="", labelled_unit_data="",
                 sector_data="", district_data="", stations = ""):

        """
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additiona .csv files containing addtional
            information on households.
        """

        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(_data_dir,
                                                'postcodes_unlabelled.csv')

        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(_data_dir,
                                              'postcodes_labelled.csv')
        if sector_data == "":
            sector_data = os.path.join(_data_dir,
                                       'sector_data.csv')

        if district_data == "":
            district_data = os.path.join(_data_dir,
                                       'district_data.csv')

        if stations == "":
            stations = os.path.join(_data_dir,
                                       'stations.csv')

        self._postcodedb = pd.read_csv(unlabelled_unit_data)
        self._population = pd.read_csv(sector_data)
        self._pets = pd.read_csv(district_data)
        self._stationsd = pd.read_csv(stations)
        self._labeldb = pd.read_csv(labelled_unit_data)
        self.num_features = ['easting', 'northing']
        self.cat_features = ['soilType']
        self.X = self._labeldb.drop(['localAuthority', 'riskLabel', 'medianPrice', 'historicallyFlooded'], axis=1)


        # continue your work here

    def train(self, models=[], update_labels="", update_hyperparameters=False):
        """Train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to train
        update_labels : str, optional
            Filename of a .csv file containing a labelled set of samples.
        tune_hyperparameters : bool, optional
            If true, models can tune their hyperparameters, where
            possible. If false, models use your chosen default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.train(fcp_methods[0])  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        """

        if update_labels:
            print("updating labelled sample file")
            # update your labelled samples

        for model in models:
            if update_hyperparameters:
                print(f"tuning {model} hyperparameters")
            else:
                print(f"training {model}")
            # Do your training for the specified models

#--------------------------------------------------------------------------

            if model == 'historic_rf':

                y = self._labeldb['historicallyFlooded']
                X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=.2, random_state=42)

                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', RobustScaler())])

                cat_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(sparse_output=False))])

                processorpipe = ColumnTransformer([
                    ('num', num_pipeline, self.num_features),
                    ('cat_imputer', cat_pipeline, self.cat_features)])

                fitpipe = Pipeline([
                    ('processorpipe', processorpipe),
                    ('rf', RandomForestClassifier())
                ])

                return fitpipe.fit(X_train, y_train)

#---------------------------------------------------------------------------


            if model == 'local_rf':

                #self.num_features.remove('elevation')
                #self.cat_features.remove('soilType')

                X = self._labeldb[['easting', 'northing']]
                y = self._labeldb['localAuthority']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestClassifier(
                        max_depth=20,
                        min_samples_leaf=1,
                        min_samples_split=2,
                        n_estimators=100,
                        random_state=42))
                ])

                return pipeline.fit(X_train, y_train)

#--------------------------------------------------------------------------
               
            if model == 'better_selection':
                #prepare data

                #changing all the input into at least have Easting, Northing, and Local Authority
                tot = self._labeldb[['easting','northing','localAuthority','riskLabel']]

                #get city's name
                city_gov_ids = self._labeldb['localAuthority'].unique()

                models={}

                for city_id in city_gov_ids:
                    # make data for each city
                    city_data = tot[tot['localAuthority'] == city_id]

                    X = city_data[['easting','northing']]
                    y = city_data['riskLabel']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
                    knn_model = KNeighborsClassifier(n_neighbors=2)
                    knn_model.fit(X_train,y_train)
                    y_pred = knn_model.predict(X_test)
                    score_knn =self.__score_test(y_pred,y_test)

                    rf_model = RandomForestClassifier()
                    rf_model.fit(X_train,y_train)
                    y_pred = rf_model.predict(X_test)
                    score_rf = self.__score_test(y_pred,y_test)

                    if score_rf > score_knn:
                        model = RandomForestClassifier()
                        model.fit(X,y)
                        models[city_id] = model
                    else:
                        model = KNeighborsClassifier(n_neighbors=2)
                        model.fit(X,y)
                        models[city_id] = model

                return models


#--------------------------------------------------------------------------

            if model == 'house_price_rf':
                copy = self._labeldb.copy()
                copy = self.add_population_to_df(copy)
                copy = self.add_pets_to_df(copy)
                copy.dropna(inplace=True)
                X = copy.drop(columns=['soilType', 'elevation', 'postcodeSector', 'postcodeDistrict', 'localAuthority',
                                       'riskLabel', 'medianPrice', 'historicallyFlooded'])
                X = X.set_index("postcode")
                y = np.log(
                    np.array(copy['medianPrice'].values))  # apply log func to make the y value in Gaussian distribute

                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=.2,
                                                                    random_state=42)
                hppipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(n_estimators=300, random_state=42))])

                return hppipe.fit(X_train, y_train)

#--------------------------------------------------------------------------

    def lookup_easting_northing(self, postcodes=None):
        """Get a dataframe of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of OSGB36 easthing and northing,
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the input unlabelled postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['M34 7QL'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL    393470	 394371
        
        >>> results = tool.lookup_easting_northing(['M34 7QL', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL  393470.0  394371.0
        AB1 2PQ       NaN       NaN
        """

        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()

        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]]

    def lookup_lat_long(self, postcodes, mode='test'):
        """Get a Pandas dataframe containing GPS latitude and longitude
        information for a collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        mode:  str
        "test" for looking up longitudes or latitudes in unlabelled data,
        "training" for looking up in labelled data


        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NAN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """
        if mode == 'training':
            frame = self._labeldb.copy()
        else:
            frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        frame['longitude'] = get_gps_lat_long_from_easting_northing(frame['easting'],
                                                                    frame['northing'])[1]
        frame['latitude'] =  get_gps_lat_long_from_easting_northing(frame['easting'],
                                                                    frame['northing'])[0]


        return frame.loc[postcodes, ['longitude', 'latitude']]
        #return pd.DataFrame(columns=['longitude', 'latitude'], index=postcodes)

#-----------------------------------------------------------------------


    def predict_flood_class_from_postcode(self, postcodes=None, method="better_selection"):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        warnings.filterwarnings("ignore")
        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()

        postcodes = self.regularize(postcodes)           

        test_set = self.__prep_unlabel()

        if method == "better_selection":

            models = self.train(flood_class_from_postcode_methods)
            condition = test_set['postcode'].isin(postcodes)

            matched_data = test_set[condition]

            results = []
            for _, row in matched_data.iterrows():
                local_authority = row['localAuthority']
                model = models[local_authority]
                prediction = model.predict(row[self.num_features].values.reshape(1, -1))

                results.append(prediction[0])

            return pd.Series(results, index=matched_data['postcode'], name='riskLabel')

        else:
            raise NotImplementedError(f"method {method} not implemented")


    def predict_flood_class_from_OSGB36_location(
        self, eastings=None, northings=None, method="better_selection"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """
        warnings.filterwarnings("ignore")


        if eastings is None:
            eastings = self._postcodedb['easting'].tolist()

        if northings is None:
            northings = self._postcodedb['northing'].tolist()

        if len(eastings) != len(northings):
            return ValueError("The shapes of eastings and northings must be the same.")

        if method == "better_selection":

            models = self.train(flood_class_from_postcode_methods)

            df = pd.DataFrame({
                    'easting': eastings,
                    'northing': northings
                })

            test_set = self.__prep_unlabel(df)


            matched_data = test_set

            results = []
            for index, row in matched_data.iterrows():
                local_authority = row['localAuthority']
                model = models[local_authority]
                prediction = model.predict(row[self.num_features].values.reshape(1, -1))

                results.append(prediction[0])

            return pd.Series(results, index=pd.MultiIndex.from_arrays([eastings, northings], names=['Easting', 'Northing']), name='riskLabel')


        else:
            raise NotImplementedError(f"method {method} not implemented")



    def predict_flood_class_from_WGS84_locations(
        self, latitudes=None, longitudes=None, method="better_selection"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """
        warnings.filterwarnings("ignore")

        if latitudes is None:
            return ValueError("Please enter an appropriate set of latitudes.")

        if longitudes is None:
            return ValueError("Please enter an appropriate set of longitudes.")

        if len(latitudes) != len(longitudes):
            return ValueError("The shapes of latitudes and longtituds must be the same.")


        eastings, northings = get_easting_northing_from_gps_lat_long(phi = latitudes, lam = longitudes)


        if method == "better_selection":

            models = self.train(flood_class_from_postcode_methods)

            df = pd.DataFrame({
                    'easting': eastings,
                    'northing': northings
                })

            test_set = self.__prep_unlabel(df)


            matched_data = test_set

            results = []
            for index, row in matched_data.iterrows():
                local_authority = row['localAuthority']
                model = models[local_authority]
                prediction = model.predict(row[self.num_features].values.reshape(1, -1))

                results.append(prediction[0])

            return pd.Series(results, index=pd.MultiIndex.from_arrays([eastings, northings], names=['Easting', 'Northing']), name='riskLabel')

        else:
            raise NotImplementedError(f"method {method} not implemented")

#-----------------------------------------------------------------------


    def predict_median_house_price(
        self, postcodes=None, method="house_price_rf"
    ):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
            
            
        Examples
        --------
        
        >>> tool = Tool()
        >>> tool.predict(median_house_price(['M34 7QL']) # doctest: +SKIP
        
        training all_england_median
        Model not recognised.
        training house_price_rf
        
        M34 7QL    165620.492167
        Name: medianPrice, dtype: float64
        
        """

        #maybe return two different things from the train set, model and full_unseen
        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()
            
        postcodes = self.regularize(postcodes)

        if method == "house_price_rf":
            model = self.train(house_price_methods)

            # compose features by searching the input postcodes in dataset
            copy = self._labeldb.copy()
            df_labelled = self.add_population_to_df(copy)
            df_labelled = self.add_pets_to_df(df_labelled)
            df_labelled.drop(columns=['soilType', 'elevation', 'localAuthority', 'riskLabel', 'medianPrice',
                                      'historicallyFlooded', 'postcodeSector', 'postcodeDistrict'], inplace=True)
            df_labelled.dropna(inplace=True)
            # df_labelled.drop_duplicates(inplace=True)

            copy = self._postcodedb.copy()
            df_unlabelled = self.add_population_to_df(copy)
            df_unlabelled = self.add_pets_to_df(df_unlabelled)
            df_unlabelled.drop(columns=['soilType', 'elevation', 'postcodeSector', 'postcodeDistrict'], inplace=True)
            df_unlabelled.dropna(inplace=True)

            # a joined dataset of labelled and unlabelled data provided
            dataset = pd.concat([df_labelled, df_unlabelled], axis=0, ignore_index=True)
            dataset.drop_duplicates(inplace=True)

            postcodes_in_dataset = dataset[dataset['postcode'].isin(postcodes)]
            df = pd.DataFrame({'postcode': postcodes})
            df = pd.merge(df, postcodes_in_dataset, on='postcode', how='left')
            features = df[['postcode', 'easting', 'northing', 'average_pop', 'avg_pets_per_household']]
            self.find_closest_region_feas(dataset, features)
            features = features.drop(columns=['postcode'])
            # full_unseen = self.add_population_to_df(self._postcodedb)
            # full_unseen = self.add_pets_to_df(self._postcodedb)
            # full_unseen = full_unseen.set_index('postcode')
            # full_unseen = full_unseen[['easting', 'northing', 'average_pop', 'avg_pets_per_household']]

            try:
                return pd.Series(np.exp(model.predict(features)), index=postcodes, name='medianPrice')

            except KeyError:
                raise KeyError('One or more of the specified postcode is not recognised.')

        else:
            raise NotImplementedError(f"method {method} not implemented")


    def predict_local_authority(
        self, eastings=None, northings=None, method="local_rf"
    ):
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            postcodes, and indexed by northing and easting.
            
            
        Examples    
        --------
        
        >>> tool = Tool()
        >>> tool.predict_local_authority() # doctest: +SKIP
        
        training do_nothing
        Model not recognised.
        training local_rf
        
        Northing  Easting
        394371    393470                        Tameside
        405669    395420                          Oldham
        289400    411900                      Birmingham
        562300    420400                       Gateshead
        296656    397726                         Walsall
                                        ...             
        264700    488200          North Northamptonshire
        434044    508055     Kingston upon Hull, City of
        432867    428431                           Leeds
        401208    462498                       Doncaster
        378785    441723           North East Derbyshire
        Length: 10000, dtype: object
        
        
        
        """

        if eastings is None:
            eastings = self._postcodedb['easting'].tolist()

        if northings is None:
            northings = self._postcodedb['northing'].tolist()

        if not isinstance(eastings, (list, np.ndarray)):
            raise ValueError("Eastings must be a list or numpy array.")
        if not isinstance(northings, (list, np.ndarray)):
            raise ValueError("Northings must be a list or numpy array.")
        if not len(eastings) == len(northings):
            raise ValueError("Northings and Eastings must be of equal length.")

        if method == "local_rf":

            model = self.train(local_authority_methods)
            
            return pd.Series(model.predict(np.column_stack((eastings, northings))), index=pd.MultiIndex.from_arrays([eastings, northings], names=['Northing', 'Easting']))

        else:
            raise NotImplementedError(f"method {method} not implemented")




    def predict_historic_flooding(
        self, postcodes=None, method="historic_rf"
    ):
        """
        
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------
        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.
            

        Returns
        -------
        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
            
            
        Examples
        --------
        >>> tool = Tool()
        >>> tool.predict_historic_flooding(['M34 7QL']) # doctest: +SKIP
        
        M34 7QL    False 
        Name: historicallyFlooded, dtype= bool
        
        
        >>> tool = Tool()
        >>> tool.predict_historic_flooding() # doctest: +SKIP
        
        M34 7QL     False
        OL4 3NQ     False
        B36 8TE     False
        NE16 3AT    False
        WS10 8DE    False
                    ...  
        NN9 7TY     False
        HU6 7YG     False
        LS12 1DY    False
        DN4 6TZ     False
        S31 9BD     False
        Name: historicallyFlooded, Length: 10000, dtype: bool
            
        """

        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()
            
        postcodes = self.regularize(postcodes)
        
        
        if method == "historic_rf":
            model = self.train(historic_flooding_methods)
            return pd.Series(model.predict(self._postcodedb.set_index(['postcode']).loc[postcodes]),
                                              index=postcodes,
                                              name='historicallyFlooded')


        else:
            raise NotImplementedError(f"method {method} not implemented")



    def predict_total_value(self, postcodes=None, geographic_level='all'):

        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
            
            
        Examples 
        --------
        
        >>> tool = Tool()
        >>> tool.predict_total_value() # doctest: +SKIP
        
                groupedPostcodes
        B1 1BT     238152.250110
        B1 1JS     246321.531282
        B1 1RD     236294.972427
        B1 1US     256154.243866
        B1 2AJ     311966.965995
                       ...      
        YO8 9LF    238285.866776
        YO8 9QT    221064.290238
        YO8 9UZ    218386.961123
        YO8 9XE    222974.077006
        YO8 9XW    208991.702800
        Name: medianPrice, Length: 10000, dtype: float64
        
        
        >>> tool = Tool()
        >>> tool.predict_total_value(['M34 7QL']) # doctest: +SKIP
        
        training house_price_rf
        
        groupedPostcodes
        M34 7QL    168554.976364
        Name: medianPrice, dtype: float64
            
        """

        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()
            
        median_prices = self.predict_median_house_price(postcodes)
        median_prices = pd.DataFrame(median_prices).reset_index()
        median_prices['groupedPostcodes'] = self.regularize(median_prices['index'], geographic_level=geographic_level)
        total_value_series = median_prices.groupby('groupedPostcodes')['medianPrice'].sum()
        return total_value_series


    def predict_annual_flood_risk(self, postcodes=None, geographic_level='all'):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.

        """
        
        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()

        risk_labels = self.predict_flood_class_from_postcode(postcodes)
        risk_labels = pd.DataFrame(risk_labels).reset_index()
        risk_labels['groupedPostcodes'] = self.regularize(risk_labels['postcode'].tolist(),geographic_level=geographic_level)
        frequent_risk = np.round(risk_labels.groupby('groupedPostcodes')['riskLabel'].mean())
        property_values = self.predict_total_value(postcodes, geographic_level=geographic_level)
        class_percent_dict = {10: 0.05,
                              9: 0.04,
                              8: 0.03,
                              7: 0.02,
                              6: 0.015,
                              5: 0.01,
                              4: 0.005,
                              3: 0.003,
                              2: 0.002,
                              1: 0.001}
        frequent_risk = frequent_risk.map(class_percent_dict)
        flood_risk = 0.05 * property_values * frequent_risk
        return flood_risk


    def collate_outputs(self, postcodes=None):

        """
        Return a pandas DataFrame containing all outputs from the training models
        (predicted median house price, local authority, historic flooding, flood class and
        annual flood risk). 
        
        Parameters
        ----------

        postcodes : list of strs
            A list containing strings of each postcode.
            

        Returns
        -------

        pandas.DataFrame
            A DataFrame containing the output predictions from all of the training models.
            

        """

        if postcodes is None:
            postcodes = self._postcodedb['postcode'].tolist()

        east_north_lookup = self.lookup_easting_northing(postcodes)

        data = {
            'median_house_price': self.predict_median_house_price(postcodes),
            'historic_flooding': self.predict_historic_flooding(postcodes),
            'flood_class': self.predict_flood_class_from_postcode(postcodes),
            'annual_flood_risk': self.predict_annual_flood_risk(postcodes, geographic_level='all')
        }


       # return pd.merge(east_north_lookup, pd.DataFrame(data), right_on='postcode', left_index=True)
        return pd.DataFrame(data)#, index=multi_index)





    def get_postcode_from_lat_long(self, lat, long):

        """
        Returns the postcode of a given latitude and longitude
        Parameters
        ----------
        lat: float
            Latitude of location
        long: float
            Longitude of location
        Returns
        -------
        str
            Postcode of location
        """
        min_lat, max_lat, min_long, max_long = self.find_min_max_lat()
        if long >= min_long and long <= max_long and lat >= min_lat and lat <= max_lat:
            easting, northing = get_easting_northing_from_gps_lat_long(lat, long)
            postcode = self.get_closest_postcode_from_easting_northing(easting, northing)
            return postcode
        else:
            print("Latitude and Longitude are not within the accepted geographic range.")
            return None

    def get_closest_postcode_from_easting_northing(self, easting, northing):
        """
        Returns the postcode of a given easting and northing
        Parameters
        ----------
        easting: float
            Easting of location
        northing: float
            Northing of location
        Returns
        -------
        str
            Postcode of location
        """

        frame = self._postcodedb.copy()
        frame['distance'] = np.sqrt((frame['easting'] - easting)**2 + (frame['northing'] - northing)**2)
        minimum = frame['distance'].idxmin()
        closest_postcode = frame.loc[minimum, 'postcode']
        return closest_postcode

    def find_min_max_lat (self):
        '''
        Returns the minimum and maximum latitude and longitude of the postcodes in the database
        Parameters
        ----------
        Returns
        -------
        float
            Minimum latitude of postcodes
            float
            Maximum latitude of postcodes
            float
            Minimum longitude of postcodes
            float
            Maximum longitude of postcodes
        '''
        frame = self._postcodedb.copy()
        frame[['latitude', 'longitude']] = frame.apply(
            lambda row: get_gps_lat_long_from_easting_northing(row['easting'], row['northing']), axis=1
        )
        return frame['latitude'].min(), frame['latitude'].max(), frame['longitude'].min(), frame['longitude'].max()


    def regularize(self, postcodes, geographic_level = 'all'):
        '''
        Returns the regularized postcode of a given postcode, based on the geographic level. 
        
        Parameters
        ----------
        postcode: list
            list of postcodes in each location
        geographic_level: str
            Geographic level of postcode to return. Options are 'all', 'area', 'district', 'sector'
        Returns
        -------
        unitlist: list
            Regularized postcodes 
        '''
        unitlist = []
        
        for postcode in postcodes:
            if geographic_level != 'all' and geographic_level != 'area' and geographic_level != 'district' and geographic_level != 'sector':
                geographic_level = 'all'
            postcode = ' '.join(postcode.split()).upper().strip()
            area = postcode[:2].replace(" ", "")
            district = postcode[2:-3].replace(" ", "")
            sector_unit = postcode[-3:].replace(" ", "")
            sector = sector_unit[0]
            unit = sector_unit[1:]
            all_but_unit = area + district + ' ' + sector

            if geographic_level == 'all':
                unitlist.append(all_but_unit + unit)
            elif geographic_level == 'area':
                unitlist.append(area)
            elif geographic_level == 'district':
                unitlist.append(area + district)
            elif geographic_level == 'sector':
                unitlist.append(all_but_unit)
        return unitlist

    def regularize_bare(self, col_to_regularize):
        '''
        Returns standard regularization. 
        Must be applied to population data column sector before matching with 
        regularized postcode sector column.

        Also use in other similar situations.

        Parameters
        ----------
        data: str
            Data to regularize
        Returns
        -------
        str
            Regularized data
        '''
        col_to_regularize = ' '.join(col_to_regularize.split()).upper().strip()
        return col_to_regularize

    def add_data(self, pop_df, labelled_df, input_label = 'postcodeSector',
                 output_label='average_pop'):
        '''
        Adds population data to the desired database.

        Parameters
        ----------
        pop_df: pandas.DataFrame
            Dataframe containing population data.
        
        labelled_df: pandas.DataFrame
            Dataframe to add population data to.
        
        input_label: str
            Label of column in pop_df to match with labelled_df.
        
        output_label: str
            Label of column in pop_df to add to labelled_df.
        
        Returns
        -------
        pandas.DataFrame
            Dataframe with population data added.
        '''

        pop_mapping = pop_df.groupby(input_label)[output_label].mean().to_dict()
        labelled_df[output_label] = labelled_df[input_label].map(pop_mapping)

        return labelled_df

    def add_population_to_df (self, dataframe_to_add_to):
        '''
        Adds population data to the desired database.

        Parameters
        ----------
        dataframe_to_add_to: pandas.DataFrame
            Dataframe to add population data to.
        Returns
        -------
        pandas.DataFrame
            Dataframe with population data added.
        '''
        dataframe_to_add_to['postcodeSector'] = self.regularize(dataframe_to_add_to['postcode'].tolist(), geographic_level='sector')
        self._population['postcodeSector'] = self._population['postcodeSector'].apply(self.regularize_bare)
        self._population['average_pop'] = (self._population['headcount'] /
                                           self._population['numberOfPostcodeUnits'])
        dataframe_to_add_to = self.add_data(self._population, dataframe_to_add_to)
        return dataframe_to_add_to

    def add_pets_to_df(self, dataframe_to_add_to):
        '''
        Adds pet data to the desired database.

        Parameters
        ----------
        dataframe_to_add_to: pandas.DataFrame
            Dataframe to add pet data to.
        
        Returns
        -------
        pandas.DataFrame
            Dataframe with pet data added.
        '''

        dataframe_to_add_to['postcodeDistrict'] = self.regularize(dataframe_to_add_to['postcode'].tolist(),
                                                                  geographic_level='district')

        self._pets['postcodeDistrict'] = self._pets['postcodeDistrict'].apply(self.regularize_bare)
        self._pets['avg_pets_per_household'] = (self._pets['catsPerHousehold'] +
                                                self._pets['dogsPerHousehold'])
        dataframe_to_add_to = self.add_data(self._pets, dataframe_to_add_to,
                                            input_label = 'postcodeDistrict',
                                            output_label = 'avg_pets_per_household')
        return dataframe_to_add_to


    def find_closest_region_feas(self, dataset, features):
        '''
        Find the closest features appearing in provided dataset for unseen postcodes

        Parameters
        ----------
        dataset: pandas.DataFrame
            Dataframe for the postcode to be searched.
        features: a sequence of input features
            The features that contain NaN values for unseen postcodes.

        Returns
        -------
        pandas.DataFrame
            Dataframe with NaN values filled with the mean value of the closest region
        '''

        nan_rows = features[features.isna().any(axis=1)]
        # Index the rows where column has NaN values
        mean_of_notnan_rows = dataset[dataset.notna().any(axis=1)].drop(columns=['postcode']).mean()
        for index, row in nan_rows.iterrows():
            postcode = row['postcode']
            district = postcode[0:-3].replace(' ', '').upper()
            unit = postcode[-3:].upper()
            postcode = district + ' ' + unit

            common_sectors = dataset[dataset['postcode'].str.startswith(postcode[0:-2])]
            common_districts = dataset[dataset['postcode'].str.startswith(postcode[0:-4])]
            if not common_sectors.empty:
                # assign the unseen postcode with the avg values of in its sector
                fea = common_sectors.drop(columns=['postcode']).mean()
            elif not common_districts.empty:
                # assign the unseen postcode with the avg values of in its district
                fea = common_districts.drop(columns=['postcode']).mean()
            else:
                fea = mean_of_notnan_rows

            fea.index = ['easting', 'northing', 'average_pop', 'avg_pets_per_household']

            if pd.isna(features.loc[index, 'easting']):
                features.loc[index,'easting'] = fea.loc['easting']
            if pd.isna(features.loc[index, 'northing']):
                features.loc[index,'northing'] = fea.loc['northing']
            if pd.isna(features.loc[index, 'average_pop']):
                features.loc[index,'average_pop'] = fea.loc['average_pop']
            if pd.isna(features.loc[index, 'avg_pets_per_household']):
                features.loc[index,'avg_pets_per_household'] = fea.loc['avg_pets_per_household']

        return features
    #-----------------------------------------------------------------------

    def get_distance_between_lat_lon(self, lat1, lon1, lat2, lon2):
        """
        Function to calculate the distance between two points. The firs point is the postcode 
        and the second point is the station.

        Parameters
        ----------
        lat1 : float
            Latitude of the postcode
        lon1 : float
            Longitude of the postcode

        lat2 : float
            Latitude of the station
        lon2 : float
            Longitude of the station

        Returns
        -------
        distance : float
            Distance between the two points in meters
        """
        #Convert latitude and longitude to easting and northing
        east_postCode, nort_postCode = get_easting_northing_from_gps_lat_long(lat1, lon1)
        east_station, nort_station = get_easting_northing_from_gps_lat_long(lat2, lon2)

        #Calculate distance between postcodes and stations
        distance = np.sqrt((east_postCode - east_station)**2 + (nort_postCode - nort_station)**2)

        return distance

    def get_closest_stations(self, postcodes, stations = None, n_stations = 1):
        """Function to return the closest n_stations to each postcode
        Parameters
        ----------

        postcodes : pandas.DataFrame
            DataFrame containing postcodes and their latitudes and longitudes
        stations : pandas.DataFrame
            DataFrame containing stations and their latitudes and longitudes
        n_stations : int
            The number of closest stations to return. Default is 1
        Returns
        -------
        closest_stations : pandas.DataFrame
            DataFrame containing the closest n_stations to each postcode in meters
        """
        # Create empty dataframe to fill with closest stations
        closest_stations = pd.DataFrame()

        if stations is None:
            stations = self._stations.copy()

        # Loop over each postcode and fill dataframe with postcode
        # and corresponding closest stations
        for postcode in postcodes['postcode'].unique():
            # Get the lat and lon of the postcode
            postcode_df = postcodes[postcodes['postcode'] == postcode]
            postcode_lat = postcode_df['latitude'].values[0]
            postcode_lon = postcode_df['longitude'].values[0]
            # Calculate the distance between the postcode and each station
            stations['distance'] = self.get_distance_between_lat_lon(postcode_lat,
                                                                     postcode_lon,
                                                                     stations['latitude'],
                                                                     stations['longitude'])
            # Sort the stations by distance and take the closest n_stations
            closest_stations = pd.concat([closest_stations,
                                          stations.sort_values(by='distance').head(n_stations)])

        closest_stations['postcode'] = postcodes['postcode']

        return closest_stations

    def __score_test(self,y_pred,Y_test):
        SCORES = np.array(
        [
            [100, 80, 60, 60, 30, 0, -30, -600, -1800, -2400],
            [80, 100, 80, 90, 60, 30, 0, -300, -1200, -1800],
            [60, 80, 100, 120, 90, 60, 30, 0, -600, -1200],
            [40, 60, 80, 150, 120, 90, 60, 300, 0, -600],
            [20, 40, 60, 120, 150, 120, 90, 600, 600, 0],
            [0, 20, 40, 90, 120, 150, 120, 900, 1200, 600],
            [-20, 0, 20, 60, 90, 120, 150, 1200, 1800, 1200],
            [-40, -20, 0, 30, 60, 90, 120, 1500, 2400, 1800],
            [-60, -40, -20, 0, 30, 60, 90, 1200, 3000, 2400],
            [-80, -60, -40, -30, 0, 30, 60, 900, 2400, 3000],
        ]
    )
        score1 = sum(
                [
                    SCORES[_p - 1, _t - 1]
                    for _p, _t in zip(y_pred, Y_test)
                ]
        )

        score2 = sum(
                [
                    SCORES[_p - 1, _t - 1]
                    for _p, _t in zip(Y_test, Y_test)
                ]
        )

        return score1/score2


    def __prep_unlabel(self, test_set = None):
        if test_set is None:
            test_set = self._postcodedb.copy()

        test_set['localAuthority'] = self.predict_local_authority(test_set['easting'].tolist(),
                                                                  test_set['northing'].tolist()).values
        return test_set


    def add_rainfall_to_df (self, dataframe_to_add_to):
        '''
        Adds rainfall data to the desired database.

        Parameters
        ----------
        dataframe_to_add_to: pandas.DataFrame
            Dataframe to add rainfall data to.
        Returns
        -------
        pandas.DataFrame
            Dataframe with rainfall data added.
        '''
        dataframe_to_add_to['latitude'], dataframe_to_add_to['longitude'] = get_gps_lat_long_from_easting_northing(dataframe_to_add_to['easting'], dataframe_to_add_to['northing'])
        dataframe_to_add_to.reset_index(inplace=True)
        dataframe_to_add_to = self.get_closest_stations(dataframe_to_add_to)
        return dataframe_to_add_to
