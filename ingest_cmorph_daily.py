import argparse
import bisect
import bz2
import calendar
from datetime import datetime
from glob import glob
import gzip
import logging
import netCDF4
import numpy as np
import os
import shutil
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
__MONTH_DAYS_NONLEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
__MONTH_DAYS_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#-----------------------------------------------------------------------------------------------------------------------
def _find_gt(sorted_values, 
             x):
    """
    Convenience function for finding the list index of the first (leftmost) value greater than x in list sorted_values.
    
    :param sorted_values: 
    :param x:
    :return: index of the first (leftmost) value greater than x in the sorted_values list
    :rtype: int
    """
    
    index = bisect.bisect_right(sorted_values, x)
    if index != len(sorted_values):
        return index
    raise ValueError

#-----------------------------------------------------------------------------------------------------------------------
def _find_lt(sorted_values, 
             x):
    """
    Convenience function for finding the list index of the first (leftmost) value greater than x in list sorted_values.
    
    :param sorted_values: 
    :param x:
    :return: index of the first (leftmost) value greater than x in the sorted_values list
    :rtype: int
    """
    
    index = bisect.bisect_left(sorted_values, x)
    if index != len(sorted_values):
        return index
    raise ValueError

# #-----------------------------------------------------------------------------------------------------------------------
# def _read_daily_cmorph_to_monthly_sum(cmorph_dir,
#                                       data_desc,
#                                       data_year,
#                                       data_month):
#     
#     # for each file in the data directory read the data and add to the cumulative
#     summed_data = np.zeros((data_desc['xdef_count'] * data_desc['ydef_count'], ))
#     for cmorph_file in os.listdir(cmorph_dir):
#         
#         # read the year and month from the file name, make sure they all match
#         file_year = int(cmorph_file[-8:-4])
#         file_month = int(cmorph_file[-4:-2])
#         if file_year != data_year:
#             continue
#         elif file_month != data_month:
#             continue
# 
#         # read the daily binary data from file, and byte swap if not little endian
#         data = np.fromfile(os.sep.join((cmorph_dir, cmorph_file)), 'f')
#         if not data_desc['little_endian']:
#             data = data.byteswap()
#             
#         # convert missing values to zero, then if missing it's not actually added when we do the summation
#         if data_desc['undef'] < 0:     # we assume the missing value is -999.0 or something like that (negative), check here
#             data[data < 0] = 0.0
#         
#         # add to the summation array
#         summed_data += data
# 
#     return summed_data

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
def _download_daily_files(destination_dir,
                          year, 
                          month,
                          obs_type='raw'):
    """
    :param destination_dir:
    :param year:
    :param month: 1 == January, ..., 12 == December   
    """

    # determine which set of days per month we'll use based on if leap year or not    
    if calendar.isleap(year):
        days_in_month = __MONTH_DAYS_LEAP
    else:
        days_in_month = __MONTH_DAYS_NONLEAP
        
    # the base URL we'll append to in order to get the individual file URLs
    year_month = str(year) + str(month).zfill(2)
    url_base = 'ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/'
    if obs_type == 'raw':
        url_base += '02_RAW/' + str(year) + '/' + year_month
    else:
        url_base += '01_GAUGE_ADJUSTED/' + str(year) + '/' + year_month
        
    # list of files we'll return
    files = []
    
    for day in range(days_in_month[month - 1]):
        
        # build the file name, URL, and local file name
        year_month_day = year_month + str(day + 1).zfill(2)
        if obs_type == 'raw':
            filename_unzipped = 'CMORPH_V1.0_RAW_0.25deg-DLY_00Z_' + year_month_day
        else:   # guage adjusted
            filename_unzipped = 'CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + year_month_day
        zip_extension = '.bz2'
        if obs_type == 'raw' and year < 2004:   # the raw files use GZIP through 2003
            zip_extension = '.gz'
        filename_zipped = filename_unzipped + zip_extension
        
        file_url  = url_base + '/' + filename_zipped
        local_filename_zipped = destination_dir + '/' + filename_zipped
        local_filename_unzipped = destination_dir + '/' + filename_unzipped
        
        _logger.info('Downloading %s', file_url)
        
        # download the zipped file
        urllib.request.urlretrieve(file_url, local_filename_zipped)

        # decompress the zipped file
        if (year >= 2004) or (obs_type == 'adjusted'):
            # use BZ2 decompression for files after 2003
            with bz2.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            # use GZIP decompression for raw files before 2004
            with gzip.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
  
        # append to our list of data files
        files.append(local_filename_unzipped)
        
        # clean up the downloaded zip file
        os.remove(local_filename_zipped)
        
    return files

#-----------------------------------------------------------------------------------------------------------------------
def _compute_days_all_leap(year_initial,
                           year_final,
                           year_since=1900):
    '''
    Computes the "number of days" equivalent for regular, incremental daily time steps given an initial year, 
    relevant/specific to time coordinates that use a 366 day calendar. Useful when using "days since <year_since>" 
    as the time units within a NetCDF dataset. The resulting list of days will represent the range of full years,
    i.e. the length of the list will be (366 * number of years). For array indices corresponding to faux Feb 29th
    elements (during non-leap years) we duplicate the previous value used for Feb 28th, and the element corresponding 
    to March 1st will be the actual Feb 28th value plus 1.
     
    :param year_initial: the initial year from which the day values should start, i.e. the first value in the output
                        array will correspond to the number of days between January 1st of this initial year and January 
                        1st of the units since year
    :param year_final: the final year through which the result values are computed
    :param year_since: the start year from which the day values are incremented, with result time steps measured
                        in days since January 1st of this year 
    :return: an array of time step increments, measured in days since midnight of January 1st of the units' "since year"
    :rtype: ndarray of ints 
    '''
    
    # arguments validation
    if year_initial < year_since:
        raise ValueError('Invalid year arguments, initial data year is before the units since year')
    elif year_final < year_initial:
        raise ValueError('Invalid year arguments, final data year is before the initial data year')

    # allocate the array of values we'll populate and return
    total_years = (year_final - year_initial + 1)
    shape = ((total_years * 366),)
    day_values = np.empty(shape, dtype=int)
    
    # find the initial number of days since the base/units year
    time_delta = datetime(year_initial + 1, 1, 1) - datetime(year_since, 1, 1)
    start_day = time_delta.days
    current_day_value = start_day
    
    for year_number, year in enumerate(range(year_initial, year_final + 1)):

        # values used within inner (day) loop
        year_start_index = year_number * 366
        leap_year = calendar.isleap(year)

        for day_of_year in range(366):
            
            # use the value of the current day count
            day_values[year_start_index + day_of_year] = current_day_value
            
            # increment the day count for all days, and back it off if it's "Feb 29th" of a non-leap year
            current_day_value += 1
            if (day_of_year == 58) and not leap_year:
                current_day_value -= 1  # back off the increment since we're now at Feb 28th and the next day is a faux Feb 29th which we'll fill with the Feb 28th value for this year 
    
    return day_values

#-----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf(cmorph_dir,
                            netcdf_file,
                            obs_type='raw',
                            download_files=True,
                            remove_files=True,
                            conus_only=False):
    """
    Ingests CMORPH daily precipitation files into a full period of record file containing daily precipitation values.
    
    :param cmorph_dir: work directory where CMORPH files are expected to be located, downloaded files will reside here
    :param netcdf_file: output NetCDF
    :param data_descriptor_file_name: file name of the data descriptor file in CMORPH directory
    :param download_files: if true then download data descriptor and data files from FTP, overwrites files in CMORPH work directory
    :param remove_files: if files were downloaded then remove them once operations have completed 
    """
    
    # read data description info into a dictionary
    data_desc = _read_description(cmorph_dir, download_files, remove_files)
    
    # get the range of years covered
    years = _get_years()
    units_since_year = 1900
    
    # get the lat/lon range limits for the full grid
    lat_start = data_desc['ydef_start']
    lat_end = data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment'])
    lon_start = data_desc['xdef_start']
    lon_end = data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment'])
    
    # generate the full range of lat/lon values
    lat_values = list(_frange(lat_start, lat_end, data_desc['ydef_increment']))
    lon_values = list(_frange(lon_start, lon_end, data_desc['xdef_increment']))

    # slice out the CONUS only lat/lon values, if called for
    if conus_only:
        
        # find lat/lon indices corresponding to CONUS bounds
        lat_start = _find_gt(lat_values, 23.0)
        lat_end = _find_lt(lat_values, 50.0) + 1
        lon_start = _find_gt(lon_values, 232.0)
        lon_end = _find_gt(lon_values, 295.0) + 1

        # get the subset of lat/lon values specific to CONUS only            
        lat_values = lat_values[lat_start : lat_end]
        lon_values = lon_values[lon_start : lon_end]

    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lat', len(lat_values))
        output_dataset.createDimension('lon', len(lon_values))
    
        output_dataset.title = data_desc['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        lat_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        lon_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        
        # set the coordinate variables' attributes
        time_variable.units = 'days since {0}-01-01'.format(units_since_year)
        lat_variable.units = 'degrees_north'
        lon_variable.units = 'degrees_east'
        time_variable.long_name = 'days (366 days per year, non-leap years have Feb. 28th duplicated as Feb 29th)'
        lat_variable.long_name = 'Latitude'
        lon_variable.long_name = 'Longitude'
        time_variable.calendar = '366_day'
        
        # compute the time values 
        time_variable[:] = _compute_days_all_leap(data_desc['start_date'].year,
                                                  years[-1], 
                                                  year_since=units_since_year)
        
            
        # longitude values are in 
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
#         lon_values = list(_frange(lon_start, lon_end, data_desc['xdef_increment']))
#         lat_values = list(_frange(lat_start, lat_end, data_desc['ydef_increment']))
        lat_variable[:] = np.array(lat_values, 'f4')
        lon_variable[:] = np.array(lon_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data_variable = output_dataset.createVariable('prcp', 
                                                      'f8', 
                                                      ('time', 'lat', 'lon',), 
                                                      fill_value=np.NaN)
        data_variable.units = 'mm'
        data_variable.standard_name = 'precipitation'
        data_variable.long_name = 'precipitation, monthly cumulative'
        data_variable.description = data_desc['title']

        # loop over each year/month, reading binary data from CMORPH files and adding into the NetCDF variable
        days_index = 0
        for year in years:
            for month in range(1, 13):

                # get the files for the month
                if download_files:
                    daily_files = _download_daily_files(cmorph_dir, year, month, obs_type)
                else:
                    suffix = str(year) + str(month).zfill(2) + '*'
                    if obs_type == 'raw':
                        filename_pattern = cmorph_dir + '/CMORPH_V1.0_RAW_0.25deg-DLY_00Z_' + suffix
                    else:   # gauge adjusted
                        filename_pattern = cmorph_dir + '/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + suffix
                        
                    daily_files = glob(filename_pattern)

                # loop over each daily file to read the data and assign it into the variable
                for day_of_month, daily_cmorph_file in enumerate(daily_files):
                    
                    # read the daily binary data from file, and byte swap if not little endian
                    data = np.fromfile(daily_cmorph_file, 'f')
                    if not data_desc['little_endian']:
                        data = data.byteswap()
            
                    # convert missing values to NaNs
                    data[data == float(data_desc['undef'])] = np.NaN
                    
                    # assume values are in lat/lon orientation
                    data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))

                    # assign into the appropriate slice for the daily time step
                    data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                    
                    days_index += 1
        
                    # if we've come to Feb 28th and it's not a leap year then duplicate the Feb 28th values as a faux Feb 29th
                    if (month == 2) and (day_of_month == 27) and not calendar.isleap(year):
                        
                        data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                        days_index += 1
                    
                # clean up, if necessary
                if remove_files:
                    for file in daily_files:
                        os.remove(file)
                    
#-----------------------------------------------------------------------------------------------------------------------
def _frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

#-----------------------------------------------------------------------------------------------------------------------
def _read_description(work_dir,
                      download_file,
                      remove_file):
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
        
    :param work_dir: directory in which the ASCII file with data description information will live temporarily 
        while this function executes, the file should be cleaned up upon successful completion
    :return: dictionary of data description keys/values
    """

    descriptor_file = os.sep.join((work_dir, 'cmorph_data_descriptor.txt'))
    
    # download the descriptor file, if necessary
    if download_file:
        
        file_url = "ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/03_PGMS/CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl"
        urllib.request.urlretrieve(file_url, descriptor_file)

    # build the data description dictionary by extracting the relevant values from the descriptor file, line by line
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
    if remove_file:
        os.remove(descriptor_file)
    
    return data_dict

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform ingest of binary CMORPH datasets to NetCDF.

    Example command line usage for reading all daily files for all months into a single NetCDF file with cumulative 
    monthly precipitation for the full period of record (all months), with all files downloaded from FTP:
    
    $ python -u ingest_cmorph.py --cmorph_dir C:/home/data/cmorph/raw \
                                 --out_file C:/home/data/cmorph_file.nc \
                                 --download_files True
                                 
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--cmorph_dir", 
                            help="Directory containing daily binary CMORPH data files for a single month", 
                            required=True)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables read from the input data", 
                            required=True)
        parser.add_argument("--download_files", 
                            help="Download data from FTP, saving files in the CMORPH data directory specified by --cmorph_dir",
                            action="store_true", 
                            default=False)
        parser.add_argument("--remove_files", 
                            help="Remove downloaded data files from the CMORPH data directory if specified by --download_files",
                            action="store_true", 
                            default=False)
        parser.add_argument("--obs_type", 
                            help="Observation type, either raw or guage adjusted",
                            choices=['raw', 'adjusted'], 
                            default='raw',
                            required=False)
        parser.add_argument("--conus_only", 
                            help="Use only continental US data (-65 through -128 degrees east, 23 through 60 degrees north)",
                            action='store_true', 
                            required=False)
        args = parser.parse_args()

        # display run info
        print('\nIngesting CMORPH precipitation dataset')
        print('Result NetCDF:   %s' % args.out_file)
        print('Work directory:  %s' % args.cmorph_dir)
        print('\n\tDownloading files:     %s' % args.download_files)
        print('\tRemoving files:        %s' % args.remove_files)
        print('\tObservation type:      %s' % args.obs_type)
        print('\tContinental US only:   %s' % args.conus_only)
        print('\nRunning...\n')
        
        # perform the ingest to NetCDF
        ingest_cmorph_to_netcdf(args.cmorph_dir,
                                args.out_file,
                                obs_type=args.obs_type,
                                download_files=args.download_files,
                                remove_files=args.remove_files,
                                conus_only=args.conus_only)

        # display the info in case the above info has scrolled past due to output from the ingest process itself
        print('\n\nSuccessfully completed')
        print('\nResult NetCDF:   %s' % args.out_file)
        print('\tObservation type:      %s\n' % args.obs_type)
        print('\tContinental US only:   %s\n' % args.conus_only)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    