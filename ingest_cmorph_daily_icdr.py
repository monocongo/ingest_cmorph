import argparse
import bisect
import bz2
import calendar
from datetime import datetime, date, timedelta
from glob import glob
import gzip
import logging
import os
import shutil
import urllib.request
import warnings

import netCDF4
import numpy as np
from pandas import date_range

# -----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------
# ignore warnings
warnings.simplefilter('ignore', Warning)

# -----------------------------------------------------------------------------------------------------------------------
# days of each calendar month, for non-leap and leap years
__MONTH_DAYS_NONLEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
__MONTH_DAYS_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# -----------------------------------------------------------------------------------------------------------------------
def _find_closest(sorted_values,
                  value,
                  before=False):
    """
    Convenience function for finding the list index of the first (leftmost) value greater than x in list sorted_values.

    :param sorted_values:
    :param value:
    :param before: if True then return the first (leftmost) value less than x in the sorted_values list, otherwise
                   return the last (rightmost) value greater than x in the sorted_values list
    :return: index of the first (leftmost) value greater than x in the sorted_values list
    :rtype: int
    """

    if before:
        index = bisect.bisect_left(sorted_values, value)
    else:
        index = bisect.bisect_right(sorted_values, value)

    if index != len(sorted_values):
        return index
    raise ValueError


# -----------------------------------------------------------------------------------------------------------------------
def _get_years():
    return list(range(2018, 2019))  # we know this, but not portable/reusable


# FIXME use the below once we work out the proxy issue on Windows
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
# -----------------------------------------------------------------------------------------------------------------------
def _get_spec_years(start_date,
                    end_date):
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    delta = end_date_obj - start_date_obj

    dates = list()

    for i in range(delta.days + 1):
        dates.append(start_date_obj + timedelta(i))

    return dates


# -----------------------------------------------------------------------------------------------------------------------
def _get_months(start_date: str,
                end_date: str):
    """

    :param str start_date: expected in "YYYY-mm-dd" format
    :param str end_date: expected in "YYYY-mm-dd" format
    :return:
    """

    # convert strings to datetimes
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    return date_range(start_date_obj, end_date_obj, freq='MS').tolist()


# -----------------------------------------------------------------------------------------------------------------------
def _download_daily_files(destination_dir: str,
                          year: int,
                          month: int,
                          obs_type='raw'):
    """
    :param str destination_dir: directory where we should download files
    :param int year: year for which we'll download all daily files
    :param int month: 1 == January, ..., 12 == December
    :param obs_type: "raw" or "adjusted"
    """

    # determine which set of days per month we'll use based on if leap year or not
    if calendar.isleap(year):
        days_in_month = __MONTH_DAYS_LEAP
    else:
        days_in_month = __MONTH_DAYS_NONLEAP

    # the base URL we'll append to in order to get the individual file URLs
    year_month = str(year) + str(month).zfill(2)
    url_base = 'https://ftp.cpc.ncep.noaa.gov/precip/'  # Changed for updates
    if obs_type == 'raw':
        url_base += 'CMORPH_V0.x/RAW/0.25deg-DLY_00Z/' + str(year) + '/' + year_month  # changed for CPC FTP
    elif obs_type == 'adjusted':
        url_base += 'CMORPH_V1.0/CRT/0.25deg-DLY_00Z/' + str(year) + '/' + year_month  # Changed for Corrected (CRT)
    else:
        url_base += 'CMORPH_RT/ICDR/0.25deg-DLY_00Z'  # changed for ICDR
        # START HERE ####################################################################################################
    # list of files we'll return
    files = []

    # list of dates for ICDR data
    # year_month_day_icdr = ['20181101', '20181102','20181103', '20181104','20181105', '20181106','20181107', '20181108','20181109', '20181110','20181111', '20181112','20181113', '20181114','20181115', '20181116','20181117', '20181118','20181119', '20181120','20181121', '20181122','20181123', '20181124','20181125', '20181126','20181127', '20181128','20181129', '20181130','20181201', '20181202','20181203', '20181204','20181205', '20181206','20181207', '20181208','20181209', '20181210','20181211', '20181212','20181213', '20181214','20181215', '20181216','20181217', '20181218','20181219', '20181220','20181221', '20181222','20181223', '20181224','20181225', '20181226','20181227', '20181228','20181229', '20181230', '20181231' ]

    for day in range(days_in_month[month - 1]):
        # build the file name, URL, and local file name
        year_month_day = year_month + str(day + 1).zfill(2)
        if obs_type == 'raw':
            filename_unzipped = 'CMORPH_V0.x_RAW_0.25deg-DLY_00Z_' + year_month_day
            # filename = 'CMORPH_V1.0_RAW_0.25deg-DLY_00Z_' + year_month_day # Changed for ICDR
        elif obs_type == 'adjusted':  # CRT
            #    filename_unzipped = 'CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + year_month_day # Changed for CRT
            filename_unzipped = 'CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + year_month_day  # Changed for CRT
        else:
            filename_unzipped = 'CMORPH_V0.x_ADJ_0.25deg-DLY_00Z_' + year_month_day  # Changed for ICDR
        if obs_type == 'raw':  # Alec - V0.x also uses .gz and year < 2020:   # the raw files use GZIP through 2003
            zip_extension = '.gz'  # for RAW
        elif obs_type == 'adjusted':
            zip_extension = '.bz2'  # for CRT
        else:
            zip_extension = '.nc'  # for ICDR
        filename_zipped = filename_unzipped + zip_extension
        # filename = filename + zip_extension

        file_url = url_base + '/' + filename_zipped
        local_filename_zipped = destination_dir + '/' + filename_zipped
        local_filename_unzipped = destination_dir + '/' + filename_unzipped
        # local_filename = destination_dir + '/' + filename

        _logger.info('Downloading %s', file_url)

        # download the zipped file
        urllib.request.urlretrieve(file_url, local_filename_zipped)

        # decompress the zipped file
        #    if (year >= 2004) or (obs_type == 'adjusted'):
        if (obs_type == 'adjusted'):  # use for V0.x RAW, which uses gzip compression
            # use BZ2 decompression for files after 2003
            with bz2.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif (obs_type == 'raw'):
            # use GZIP decompression for raw files before 2004
            with gzip.open(local_filename_zipped, 'r') as f_in, open(local_filename_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            local_filename_unzipped = local_filename_zipped

        # append to our list of data files
        files.append(local_filename_unzipped)
        # files.append(local_filename)

        # clean up the downloaded zip file
        # os.remove(local_filename_zipped)

    return files


# -----------------------------------------------------------------------------------------------------------------------
def _compute_days_full_years(year_initial,
                             year_final,
                             year_since=1900,
                             month_initial=1,
                             day_initial=1,
                             month_final=12,
                             day_final=31):
    '''
    Computes the "number of days" equivalent for regular, incremental daily time steps given an initial year.
    Useful when using "days since <year_since>" as the time units within a NetCDF dataset. The resulting list
    of days will represent the range of full years, i.e. from January 1st of the initial year through December 31st
    of the final year.

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

    # datetime objects from the years
    date_initial = datetime(year_initial, month_initial,
                            day_initial)  # assumes start at beginning of year (issue with partial records like Real-time ICDR)
    date_final = datetime(year_final, month_final,
                          day_final)  # assumes end of year, issue with partial records like real-time ICDR
    date_since = datetime(year_since, 1, 1)

    # starting day value, i.e. first number of days since the time units' "since year"
    days_initial = (date_initial - date_since).days

    # total number of days between Jan 1st of the initial year and Dec 31st of the final year
    total_days = (date_final - date_initial).days + 1

    # list of day values starting at the initial number of days since the time units' "since year"
    days_since = range(days_initial, days_initial + total_days)

    return np.array(days_since)


# -----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf(cmorph_dir: str,
                            netcdf_file: str,
                            start_date: str,
                            end_date: str,
                            obs_type='raw',
                            download_files=True,
                            remove_files=True,
                            conus_only=False,
                            manual_dates=False):
    """
    Ingests CMORPH daily precipitation files into a full period of record file containing daily precipitation values.

    :param str cmorph_dir: work directory where CMORPH files are expected to be
        located, downloaded files will reside here
    :param str netcdf_file: output NetCDF file path
    :param str start_date: expected in "YYYY-mm-dd" format
    :param str end_date: expected in "YYYY-mm-dd" format
    :param obs_type: "raw" or "adjusted"
    :param download_files: if true then download the data descriptor and data files from FTP, overwrites files in CMORPH work directory
    :param remove_files: if files were downloaded then remove them once operations have completed
    :param conus_only: ingest only data for CONUS
    :param manual_dates:
    :return:
    """

    # read data description info into a dictionary
    data_desc = _read_description(cmorph_dir, download_files, remove_files, obs_type)

    lat_start = data_desc['ydef_start']
    lat_end = data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment'])
    lon_start = data_desc['xdef_start']
    lon_end = data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment'])

    # generate the full range of lat/lon values
    lat_values = list(_frange(lat_start, lat_end, data_desc['ydef_increment']))
    lon_values = list(_frange(lon_start, lon_end, data_desc['xdef_increment']))

    # get length of lat and lon values for inserting data slices
    lat_len = len(lat_values)
    lon_len = len(lon_values)

    # slice out the CONUS lat/lon values, if called for
    if conus_only:
        # find lat/lon indices corresponding to CONUS bounds
        lat_start = _find_closest(lat_values, 23.0)
        lat_end = _find_closest(lat_values, 50.0) + 1
        lon_start = _find_closest(lon_values, 232.0)
        lon_end = _find_closest(lon_values, 295.0) + 1

        # get the subset of lat/lon values specific to CONUS only
        lat_values = lat_values[lat_start: lat_end]
        lon_values = lon_values[lon_start: lon_end]

    if manual_dates:

        # get range of dates to cover
        dates = _get_months(start_date, end_date)

    else:
        # get the range of years covered
        years = _get_years()

    units_since_year = 1900

    # get the lat/lon range limits for the full grid
    # create a corresponding NetCDF
    # The opening and closing of this file could be causing I/O errors - Move to __main__ function, keep open
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:

        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lat', len(lat_values))
        output_dataset.createDimension('lon', len(lon_values))

        # global attributes
        output_dataset.title = data_desc['title']

        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        lat_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        lon_variable = output_dataset.createVariable('lon', 'f4', ('lon',))

        # set the coordinate variables' attributes
        time_variable.units = 'days since {0}-01-01'.format(units_since_year)
        lat_variable.units = 'degrees_north'
        lon_variable.units = 'degrees_east'
        time_variable.long_name = 'Time'
        lat_variable.long_name = 'Latitude'
        lon_variable.long_name = 'Longitude'
        time_variable.calendar = 'gregorian'

        # set the coordinate variable values
        if manual_dates:
            time_variable[:] = _compute_days_full_years(dates[0].year,
                                                        dates[len(dates) - 1].year,
                                                        units_since_year,
                                                        dates[0].month,
                                                        dates[0].day,
                                                        dates[len(dates) - 1].month,
                                                        dates[len(dates) - 1].day)
        else:
            time_variable[:] = _compute_days_full_years(data_desc['start_date'].year,
                                                        years[-1],
                                                        year_since=units_since_year)

        lat_variable[:] = np.array(lat_values, 'f4')
        lon_variable[:] = np.array(lon_values, 'f4')

        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data_variable = output_dataset.createVariable('prcp',
                                                      'f4',
                                                      ('time', 'lat', 'lon',),
                                                      fill_value=np.NaN)
        data_variable.units = 'mm'
        data_variable.standard_name = 'precipitation'
        data_variable.long_name = 'Precipitation'
        data_variable.description = data_desc['title']

        # loop over each year/month, reading binary data from CMORPH files and adding into the NetCDF variable
        # NEED TO MAKE THIS IF AND ONLY UPDATE WITH MANUAL_DATES OPTION
        days_index = 0

        # Added for manual dates
        if manual_dates:
            for datee in dates:
                if download_files:
                    daily_files = _download_daily_files(cmorph_dir, datee.year, datee.month, obs_type)
                else:
                    suffix = str(datee.year) + str(datee.month).zfill(2) + '*'
                    if obs_type == 'raw':
                        filename_pattern = cmorph_dir + '/CMORPH_V0.x_RAW_0.25deg-DLY_00Z_' + suffix  # for RAW
                    elif obs_type == 'adjusted':  # gauge adjusted
                        filename_pattern = cmorph_dir + '/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + suffix  # for CRT ('adjusted' argument)
                    else:
                        filename_pattern = cmorph_dir + '/CMORPH_V0.x_ADJ_0.25deg-DLY_00Z_' + suffix  # for ICDR
                    daily_files = glob(filename_pattern)  # can we assume sorted in date ascending order?
                    _logger.info(daily_files[0])
                    print(len(daily_files))

                for daily_cmorph_file in daily_files:
                    if not obs_type == 'icdr':

                        # read the daily binary data from file, and byte swap if not little endian
                        data = np.fromfile(daily_cmorph_file, 'f')
                        if not data_desc['little_endian']:
                            data = data.byteswap()

                        # convert missing values to NaNs
                        data[data == float(data_desc['undef'])] = np.NaN

                        # assume values are in lat/lon orientation
                        data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))

                        # assign into the appropriate slice for the daily time step
                        #    _logger.info('days_index: %s', days_index)
                        #    _logger.info('lat_start/lat_end: %s', lat_start)
                        #    _logger.info('lat_end: %s', lat_end)
                        #    _logger.info('lon_start: %s', lon_start)
                        #    _logger.info('lon_end: %s', lon_end)
                        #   data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                        data_variable[days_index, :, :] = data[:, : lat_len, : lon_len]  # tweaked to use index values

                        days_index += 1
                    else:
                        # read data from ICDR netcdf file
                        dataset = netCDF4.Dataset(daily_cmorph_file, mode='r')
                        data = np.array(dataset.variables['cmorph'])

                        # convert missing values to NaNs
                        data[data == float(data_desc['undef'])] = np.NaN

                        # assume values are in lat/lon orientation
                        data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))

                        # assign into the appropriate slice for the daily time step
                        #    _logger.info('days_index: %s', days_index)
                        #    _logger.info('lat_start/lat_end: %s', lat_start)
                        #    _logger.info('lat_end: %s', lat_end)
                        #    _logger.info('lon_start: %s', lon_start)
                        #    _logger.info('lon_end: %s', lon_end)
                        #   data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                        data_variable[days_index, :, :] = data[:, : lat_len, : lon_len]  # tweaked to use index values

                        days_index += 1
                # clean up, if necessary
                if remove_files:
                    for file in daily_files:
                        os.remove(file)
        else:
            # Original
            for year in years:

                # FIXME debug only -- remove
                pass

                for month in range(1, 13):

                    # FIXME debug only -- remove
                    pass

                    # get the files for the month
                    if download_files:
                        daily_files = _download_daily_files(cmorph_dir, year, month, obs_type)
                    else:
                        suffix = str(year) + str(month).zfill(2) + '*'
                        if obs_type == 'raw':
                            filename_pattern = cmorph_dir + '/CMORPH_V0.x_RAW_0.25deg-DLY_00Z_' + suffix  # for RAW
                        elif obs_type == 'adjusted':  # gauge adjusted
                            filename_pattern = cmorph_dir + '/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_' + suffix  # for CRT ('adjusted' argument)
                        else:
                            filename_pattern = cmorph_dir + '/CMORPH_V0.x_ADJ_0.25deg-DLY_00Z_' + suffix  # for ICDR
                        daily_files = glob(filename_pattern)  # can we assume sorted in date ascending order?
                        _logger.info(daily_files[0])
                        print(len(daily_files))

                    # loop over each daily file to read the data and assign it into the variable
                    for daily_cmorph_file in daily_files:
                        if not obs_type == 'icdr':

                            # read the daily binary data from file, and byte swap if not little endian
                            data = np.fromfile(daily_cmorph_file, 'f')
                            if not data_desc['little_endian']:
                                data = data.byteswap()

                            # convert missing values to NaNs
                            data[data == float(data_desc['undef'])] = np.NaN

                            # assume values are in lat/lon orientation
                            data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))

                            # assign into the appropriate slice for the daily time step
                            #    _logger.info('days_index: %s', days_index)
                            #    _logger.info('lat_start/lat_end: %s', lat_start)
                            #    _logger.info('lat_end: %s', lat_end)
                            #    _logger.info('lon_start: %s', lon_start)
                            #    _logger.info('lon_end: %s', lon_end)
                            #   data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                            data_variable[days_index, :, :] = data[:, : lat_len,
                                                              : lon_len]  # tweaked to use index values

                            days_index += 1
                        else:
                            # read data from ICDR netcdf file
                            dataset = netCDF4.Dataset(daily_cmorph_file, mode='r')
                            data = np.array(dataset.variables['cmorph'])

                            _logger.info("Initial data shape for the day: %s", data.shape)

                            # convert missing values to NaNs
                            data[data == float(data_desc['undef'])] = np.NaN

                            # assume values are in lat/lon orientation
                            data = np.reshape(data, (1, data_desc['ydef_count'], data_desc['xdef_count']))

                            _logger.info("Data shape post reshape: %s", data.shape)

                            # assign into the appropriate slice for the daily time step
                            #    _logger.info('days_index: %s', days_index)
                            #    _logger.info('lat_start/lat_end: %s', lat_start)
                            #    _logger.info('lat_end: %s', lat_end)
                            #    _logger.info('lon_start: %s', lon_start)
                            #    _logger.info('lon_end: %s', lon_end)
                            #   data_variable[days_index, :, :] = data[:, lat_start : lat_end, lon_start : lon_end]
                            data_variable[days_index, :, :] = data[:, : lat_len,
                                                              : lon_len]  # tweaked to use index values

                            days_index += 1

                    # clean up, if necessary
                    if remove_files:
                        for file in daily_files:
                            os.remove(file)


# -----------------------------------------------------------------------------------------------------------------------
def _frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


# -----------------------------------------------------------------------------------------------------------------------
def _read_description(work_dir,
                      download_file,
                      remove_file, obs_type='raw'):
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
    #    if download_file:

    #        file_url = "ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/03_PGMS/CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl"
    #        urllib.request.urlretrieve(file_url, descriptor_file)

    if download_file:
        if obs_type == 'raw':
            file_url = "https://ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/CTL/CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl"  # Changed from Oliiver FTP James used to have
            urllib.request.urlretrieve(file_url, descriptor_file)
        else:
            file_url = "https://ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/CTL/CMORPH_V1.0_CRT_0.25deg-3HLY.ctl"  # switch to daily?
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
                if obs_type == 'adjusted' or obs_type == 'icdr':
                    data_dict['start_date'] = datetime.strptime(words[3][3:],
                                                                '%d%b%Y')  # for CRT adjusted example: "00zjan1998"
                else:
                    data_dict['start_date'] = datetime.strptime(words[3], '%d%b%Y')  # example: "01jan1998"
            elif words[0] == 'OPTIONS':
                if words[2] == 'big_endian':
                    data_dict['little_endian'] = False
                else:  # assume words[2] == 'little_endian'
                    data_dict['little_endian'] = True
            elif words[
                0] == 'cmorph':  # looking for a line like this: "cmorph   1   99 yyyyy CMORPH Version 1.o daily precipitation (mm)"
                data_dict['variable_description'] = ' '.join(words[4:])
            elif words[0] == 'TITLE':
                data_dict['title'] = ' '.join(words[1:])
    # Add a nested "if, else" under line 401 to change 3-hly to daily & mm/3hr to mm for CRT data?
    # clean up
    if remove_file:
        os.remove(descriptor_file)

    return data_dict


# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform ingest of binary CMORPH datasets to NetCDF.

    Example command line usage for reading all daily files for all months into a single NetCDF file containing gauge 
    adjusted daily precipitation for the full period of record (all months), with all files downloaded from FTP and 
    left in place:

    $ python -u ingest_cmorph.py --cmorph_dir /data/cmorph/raw \
                                 --out_file C:/home/data/cmorph_file.nc \
                                 --download --obs_type adjusted

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
        parser.add_argument("--download",
                            help="Download data from FTP, saving files in the CMORPH data directory specified by --cmorph_dir",
                            action="store_true",
                            default=False)
        parser.add_argument("--clean_up",
                            help="Remove downloaded data files from the CMORPH data directory if specified by --download",
                            action="store_true",
                            default=False)
        parser.add_argument("--obs_type",
                            help="Observation type, either raw or gauge adjusted",
                            choices=['raw', 'adjusted', 'icdr'],
                            default='adjusted',
                            required=False)
        parser.add_argument("--conus",
                            help="Use only continental US data (-65 through -128 degrees east, 23 through 60 degrees north)",
                            action='store_true',
                            required=False)
        # Added to allow user specified start-/end-dates for ICDR real-time data and specific data periods
        parser.add_argument("--manual_dates",
                            help="Uses user specified start and end date arguments (--start_date & --end_date respectively)",
                            action="store_true",
                            default=False)
        parser.add_argument("--start_date",
                            help="Start date (YYYY-MM-DD format) for data being downloaded if not downloading a period of record starting on Jan. 1, 1998",
                            required=False)
        parser.add_argument("--end_date",
                            help="End date (YYYY-MM-DD format) for data being downloaded if not downloading a period of record ending on Dec. 31, 2017",
                            required=False)
        args = parser.parse_args()

        # display run info
        print('\nIngesting CMORPH precipitation dataset')
        print('Result NetCDF:   %s' % args.out_file)
        print('Work directory:  %s' % args.cmorph_dir)
        print('\n\tDownloading files:     %s' % args.download)
        print('\tRemoving files:        %s' % args.clean_up)
        print('\tObservation type:      %s' % args.obs_type)
        print('\tContinental US only:   %s' % args.conus)
        print('\nRunning...\n')

        # perform the ingest to NetCDF
        ingest_cmorph_to_netcdf(args.cmorph_dir,
                                args.out_file,
                                start_date=args.start_date,
                                end_date=args.end_date,
                                obs_type=args.obs_type,
                                download_files=args.download,
                                remove_files=args.clean_up,
                                conus_only=args.conus,
                                manual_dates=args.manual_dates)

        # display the info in case the above info has scrolled past due to output from the ingest process itself
        print('\nSuccessfully completed')
        print('\nResult NetCDF:   %s\n' % args.out_file)
        print('\tObservation type:      %s' % args.obs_type)
        print('\tContinental US only:   %s\n' % args.conus)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
