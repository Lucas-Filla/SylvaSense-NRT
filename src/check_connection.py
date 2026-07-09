import ee

def verify_connection():
    print('\nInitializing link...')

    
    try:
         #initializing the library
        ee.Initialize()
        #Porto Velho, Brazil
        test_point = ee.Geometry.Point([-63.90, -8.76]) #longitude, latitude (I didn't know it was supposed to be switched at first which led to some issues)
        #This defines the data that I want
        #COPERNICUS/S2: Accessing the Sentinel-2 Satellites
        #SR: Surface Reflectance - Means that the data has been preprocessed to remove the haze in the atmospher
        #HARMONIZED: Harmonizes the images from the satellite so the model doesn't get confused
        image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(test_point) \
            .sort('system:time_start', False) \
            .first()
        
        #Using getInfo because its just a connection check
        image_id = image.get('system:index').getInfo()
        date = image.date().format('YYYY-MM-DD').getInfo()
        cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

        print("\nConnection Successful!")
        print("Satellite: Sentinel-2")
        print(f"Image ID: {image_id}")
        print(f"Capture Date: {date}")
        print(f"Cloud Cover: {cloud_cover}")
        print("\n We are officially connected with the Sentinel Constellation!\n")
        
    except Exception as e:
        print("\nConnection Failed :(")
        print(f"Error: {e}")
        print("\n Make sure you ran 'earthengine set_project YOUR_PROJECT_ID' after authenticating.\n")

if __name__ == "__main__":
    verify_connection()
