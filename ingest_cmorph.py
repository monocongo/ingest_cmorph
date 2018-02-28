import argparse
import bz2
import calendar
from datetime import datetime
# import ftplib
import gzip
import logging
import netCDF4
import numpy as np
import os
import shutil
import urllib.error
import urllib.request
import warnings

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# ignore warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
# days of each calendar month, for non-leap and leap years
_MONTH_DAYS_NONLEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_DAYS_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#-----------------------------------------------------------------------------------------------------------------------
def _read_daily_cmorph_to_monthly_sum(cmorph_files,
                                      data_desc,
                                      data_year,
                                      data_month):
    
    # for each file in the data directory read the data and add to the cumulative
    summed_data = np.zeros((data_desc['xdef_count'] * data_desc['ydef_count'], ))
    for cmorph_file in cmorph_files:
        
        # read the year and month from the file name, make sure they all match
        file_year = int(cmorph_file[-8:-4])
        file_month = int(cmorph_file[-4:-2])
        if file_year != data_year:
            continue
        elif file_month != data_month:
            continue

        # read the daily binary data from file, byte swap if not little endian, and mask the missing/fill values
        data = np.fromfile(cmorph_file, 'f')
        if not data_desc['little_endian']:
            data = data.byteswap()
            
        # replace missing values with zeros, so when we sum with previous values 
        # we're not adding anything to the sum where we actually have missing data
        data[data == data_desc['undef']] = 0.0
        
        # add to the summation array
        summed_data += data

    return summed_data

#-----------------------------------------------------------------------------------------------------------------------
def _get_years():
    
    return list(range(1998, 2018))  # we know this, but not portable/reusable

#FIXME use the below once we work out the proxy issue on Windows
#
#     # read the listing of directories from the list of raw data years, these should all be 4-digit years
#     f = ftplib.FTP()
#     f.connect('ftp://filsrv.cicsnc.org')
#     f.login('anonymous')
#     f.cwd('olivier/data_CMORPH_NIDIS/02_RAW')
#     ls = f.mlsd()
#     f.close()
# 
#     years = []
#     for items in ls:
#         if item['type'] == 'dir':
#             year = item['name']
#             if year.isdigit() and len(year) == 4 and int(year) > 1900:
#                 years.append(year)
#             
#     return years

#-----------------------------------------------------------------------------------------------------------------------
def _download_data_descriptor(work_dir):
    
    file_url = "ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/03_PGMS/CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl"
    data_descriptor_file = work_dir + '/cmorph_data_descriptor.txt'
    urllib.request.urlretrieve(file_url, data_descriptor_file)
    return data_descriptor_file
    
#-----------------------------------------------------------------------------------------------------------------------
def _download_daily_files(destination_dir,
                          year, 
                          month,
                          raw=True):
    """
    Downloads the daily files corresponding to a specific month.
    
    :param destination_dir: location where downloaded files will reside
    :param year:
    :param month: 1 == January, ..., 12 == December
    :param raw: True: ingest raw data files, False: ingest the gauge adjusted data files
    :return: list of the downloaded files (full paths)
    """

    # determine which set of days per month we'll use based on if leap year or not    
    if calendar.isleap(year):
        days_in_month = _MONTH_DAYS_LEAP
    else:
        days_in_month = _MONTH_DAYS_NONLEAP
        
    # base URL we'll append to to get the individual file URLs
    year_month = str(year) + str(month).zfill(2)
    if raw:
        url_base = 'ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/02_RAW/' + str(year) + '/' + year_month
        filename_base = 'CMORPH_V1.0_RAW_0.25deg-DLY_00Z_'
    else:
        url_base = 'ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/01_GAUGE_ADJUSTED/' + str(year) + '/' + year_month
        filename_base = 'CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_'

    # list of files we'll return
    files = []
    
    for day in range(days_in_month[month - 1]):
        
        # build the file name, URL, and local file name
        filename_unzipped = filename_base + year_month + str(day + 1).zfill(2)
        zip_extension = '.gz'
        if not raw or year >= 2004:   # after 2003 the RAW data uses bz2, all gauge adjusted files use bz2
            zip_extension = '.bz2'
        filename_zipped = filename_unzipped + zip_extension
        
        file_url  = url_base + '/' + filename_zipped
        local_filename_zipped = destination_dir + '/' + filename_zipped
        local_filename_unzipped = destination_dir + '/' + filename_unzipped
        
        _logger.info('Downloading %s', file_url)
        
        try:
            # download the zipped file
            urllib.request.urlretrieve(file_url, local_filename_zipped)
    
            # decompress the zipped file
            if not raw or year >= 2004:
                # use BZ2 decompression for all gauge adjusted files and RAW files after 2003
                with bz2.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                # use BZ2 decompression for files before 2004
                with gzip.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
      
            # append to our list of data files
            files.append(local_filename_unzipped)
            
            # clean up the downloaded zip file
            os.remove(local_filename_zipped)
        
        except urllib.error.URLError:
        
            # download failed, move to next
            continue

    return files

#-----------------------------------------------------------------------------------------------------------------------
def _compute_days(initial_year,
                  total_months,
                  initial_month=1,
                  units_start_year=1800):
    '''
    Computes the "number of days" equivalent for regular, incremental monthly time steps given an initial year/month.
    Useful when using "days since <start_date>" as time units within a NetCDF dataset.
    
    :param initial_year: the initial year from which the day values should start, i.e. the first value in the output
                        array will correspond to the number of days between January of this initial year since January 
                        of the units start year
    :param total_months: the total number of monthly increments (time steps measured in days) to be computed
    :param initial_month: the month within the initial year from which the day values should start, with 1: January, 2: February, etc.
    :param units_start_year: the start year from which the monthly increments are computed, with time steps measured
                             in days since January of this starting year 
    :return: an array of time step increments, measured in days since midnight of January 1st of the units start year
    :rtype: ndarray of ints 
    '''

    # compute an offset from which the day values should begin 
    start_date = datetime(units_start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)
    
    # loop over all time steps (months)
    for i in range(total_months):
        
        years = int((i + initial_month - 1) / 12)   # the number of years since the initial year 
        months = int((i + initial_month - 1) % 12)  # the number of months since January
        
        # cook up a datetime object for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)
        
        # get the number of days since the initial date
        days[i] = (current_date - start_date).days
    
    return days

#-----------------------------------------------------------------------------------------------------------------------
def _init_netcdf(netcdf_file,
                 work_dir):
    """
    Initializes the NetCDF that will be written by the ASCII to NetCDF ingest process.
    
    :param netcdf_file: output NetCDF we're initializing
    :param work_dir: directory where files file name of the data descriptor file in CMORPH directory
    """
    
    # read data description info
    data_desc = _read_description(work_dir)
    
    # get the years covered
    years = _get_years()
        
    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lon', data_desc['xdef_count'])
        output_dataset.createDimension('lat', data_desc['ydef_count'])
    
        #TODO provide additional attributes for CF compliance, data discoverability, etc.
        output_dataset.title = data_desc['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        x_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        y_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        
        # set the coordinate variables' attributes
        data_desc['units_since_year'] = 1800
        time_variable.units = 'days since %s-01-01 00:00:00' % data_desc['units_since_year']
        x_variable.units = 'degrees_east'
        y_variable.units = 'degrees_north'
        
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
        lon_values = list(_frange(data_desc['xdef_start'], data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment']), data_desc['xdef_increment']))
        lat_values = list(_frange(data_desc['ydef_start'], data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment']), data_desc['ydef_increment']))
        x_variable[:] = np.array(lon_values, 'f4')
        y_variable[:] = np.array(lat_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data_variable = output_dataset.createVariable('prcp', 
                                                      'f8', 
                                                      ('time', 'lat', 'lon',), 
                                                      fill_value=np.NaN)

        # variable attributes
        data_variable.units = 'mm'
        data_variable.standard_name = 'precipitation'
        data_variable.long_name = 'precipitation, monthly cumulative'
        data_variable.description = data_desc['title']

    return data_desc

#-----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf_full(work_dir,
                                 netcdf_file,
                                 raw=True):
    """
    Ingests CMORPH daily precipitation files into a full period of record file containing monthly cumulative precipitation.
    
    :param work_dir: work directory where downloaded CMORPH files will temporarily reside while being used for ingest
    :param netcdf_file: output NetCDF
    :param raw: if True then ingest from raw files, otherwise ingest from adjusted/corrected files 
    """
    
    # create/initialize the NetCDF dataset, get back a data descriptor dictionary
    data_desc = _init_netcdf(netcdf_file, work_dir)

    with netCDF4.Dataset(netcdf_file, 'a') as output_dataset:
    
        # compute the time values 
        total_years = 2017 - int(data_desc['start_date'].year) + 1   #FIXME replace this hard-coded value with an additional end_year entry in the data_desc
        output_dataset.variables['time'][:] = _compute_days(data_desc['start_date'].year,
                                                            total_years * 12,  
                                                            initial_month=data_desc['start_date'].month,
                                                            units_start_year=data_desc['units_since_year'])
        
        # get a handle to the precipitation variable, for convenience
        data_variable = output_dataset.variables['prcp']
        
        # loop over each year/month, reading binary data from CMORPH files and adding into the NetCDF variable
        for year in range(data_desc['start_date'].year, 2018):  # from start year through 2017, replace the value 2018 here with some other method of determining this value from the dataset itself
            for month in range(1, 13):

                # get the files for the month
                downloaded_files = _download_daily_files(work_dir, year, month, raw)
                       
                if len(downloaded_files) > 0:

                    # read all the data for the month as a sum from the daily values, assign into the appropriate slice of the variable
                    data = _read_daily_cmorph_to_monthly_sum(downloaded_files, data_desc, year, month)
                    
                    # assume values are in lat/lon orientation
                    data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))
    
                    # get the time index, which is actually the month's count from the start of the period of record                
                    time_index = ((year - data_desc['start_date'].year) * 12) + month - 1
                    
                    # assign into the appropriate slice for the monthly time step
                    data_variable[time_index, :, :] = data
            
                    # clean up
                    for file in downloaded_files:
                        os.remove(file)
                    
#-----------------------------------------------------------------------------------------------------------------------
def _frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

#-----------------------------------------------------------------------------------------------------------------------
def _read_description(work_dir):
    """
    Reads a data descriptor file, example below:
    
        DSET ../0.25deg-DLY_00Z/%y4/%y4%m2/CMORPH_V1.0_RAW_0.25deg-DLY_00Z_%y4%m2%d2  
        TITLE  CMORPH Version 1.0BETA Version, daily precip from 00Z-24Z 
        OPTIONS template little_endian
        UNDEF  -999.0
        XDEF 1440 LINEAR    0.125  0.25
        YDEF  480 LINEAR  -59.875  0.25
        ZDEF   01 LEVELS 1
        TDEF 99999 LINEAR  01jan1998 1dy 
        VARS 1
        cmorph   1   99 yyyyy CMORPH Version 1.o daily precipitation (mm)  
        ENDVARS
        
    :param descriptor_file: ASCII file with data description information
    :return: dictionary of data description keys/values
    """
    
    descriptor_file = _download_data_descriptor(work_dir)
    
    data_dict = {}    
    with open(descriptor_file, 'r') as fp:
        for line in fp:
            words = line.split()
            if words[0] == 'UNDEF':
                data_dict['undef'] = float(words[1])
            elif words[0] == 'XDEF':
                data_dict['xdef_count'] = int(words[1])
                data_dict['xdef_start'] = float(words[3])
                data_dict['xdef_increment'] = float(words[4])
            elif words[0] == 'YDEF':
                data_dict['ydef_count'] = int(words[1])
                data_dict['ydef_start'] = float(words[3])
                data_dict['ydef_increment'] = float(words[4])
            elif words[0] == 'TDEF':
                data_dict['start_date'] = datetime.strptime(words[3], '%d%b%Y')  # example: "01jan1998"
            elif words[0] == 'OPTIONS':
                if words[2] == 'big_endian':
                    data_dict['little_endian'] = False
                else:   # assume words[2] == 'little_endian'
                    data_dict['little_endian'] = True
            elif words[0] == 'cmorph':  # looking for a line like this: "cmorph   1   99 yyyyy CMORPH Version 1.o daily precipitation (mm)"
                data_dict['variable_description'] = ' '.join(words[4:])
            elif words[0] == 'TITLE':
                data_dict['title'] = ' '.join(words[1:])

    # clean up
    os.remove(descriptor_file)

    return data_dict

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform ingest of binary CMORPH datasets to NetCDF.

    Example command line usage for reading all daily files for all months into a single NetCDF file with cumulative 
    monthly precipitation for the full period of record (all months), with all files downloaded from FTP and removed 
    once processing completes, for gauge adjusted data:
    
    $ python -u ingest_cmorph.py --cmorph_dir C:/home/data/cmorph/raw \
                                 --out_file C:/home/data/cmorph_file.nc \
                                 --adjusted
                                 
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--work_dir", 
                            help="Directory where CMORPH daily files will be downloaded before being ingested to NetCDF", 
                            required=True)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables read from the input data", 
                            required=True)
        feature_parser = parser.add_mutually_exclusive_group(required=False)
        feature_parser.add_argument('--raw', 
                                    dest='feature', 
                                    action='store_true')
        feature_parser.add_argument('--adjusted', 
                                    dest='feature', 
                                    action='store_false')
        feature_parser.set_defaults(feature=True)
        args = parser.parse_args()

        # perform the ingest to NetCDF
        ingest_cmorph_to_netcdf_full(args.work_dir,
                                     args.out_file,
                                     raw=args.feature)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    