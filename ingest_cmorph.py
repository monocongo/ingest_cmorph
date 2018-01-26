import argparse
from datetime import datetime
import ftplib
import logging
import netCDF4
import numpy as np
import os
import urllib
import warnings
import calendar

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
def _read_daily_cmorph_to_monthly_sum(cmorph_dir,
                                      data_desc,
                                      data_year,
                                      data_month):
    
    # for each file in the data directory read the data and add to the cumulative
    summed_data = np.zeros((data_desc['xdef_count'] * data_desc['ydef_count'], ))
    for cmorph_file in os.listdir(cmorph_dir):
        
        # read the year and month from the file name, make sure they all match
        file_year = cmorph_file[-8:-4]
        file_month = cmorph_file[-4:-2]
        if file_year != data_year:
            continue
        elif file_month != data_month:
            continue

        # read the daily binary data from file, byte swap if not little endian, and mask the missing/fill values
        data = np.fromfile(os.sep.join((cmorph_dir, cmorph_file)), 'f')
        if not data_desc['little_endian']:
            data = data.byteswap()
        data = np.ma.masked_values(data, data_desc['undef'])
            
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
def _download_data_descriptor(data_descriptor_file):
    
    file_url = "ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/03_PGMS/CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl"
    urllib.request.urlretrieve(file_url, data_descriptor_file)
    
#-----------------------------------------------------------------------------------------------------------------------
def _download_daily_files(destination_dir,
                          year, 
                          month):
    """
    :param destination_dir:
    :param year:
    :param month: 1 == January, ..., 12 == December   
    """
    
    #TODO try using urllib.request.urlretrieve() instead of ftplib
    if calendar.isleap(year):
        days_in_month = _MONTH_DAYS_LEAP
    else:
        days_in_month = _MONTH_DAYS_NONLEAP
        
    # base URL we'll append to to get the individual file URLs
    year_month = str(year) + str(month).zfill(2)
    url_base = 'ftp://filsrv.cicsnc.org/olivier/data_CMORPH_NIDIS/02_RAW/' + str(year) + '/' + year_month

    # list of files we'll return
    files = []
    
    for day in range(days_in_month[month - 1]):
        file_name = 'CMORPH_V1.0_RAW_0.25deg-DLY_00Z_' + year_month + str(day + 1).zfill(2) + '.gz' 
        file_url  = url_base + '/' + file_name
        local_filename = destination_dir + '/' + file_name
        urllib.request.urlretrieve(file_url, local_filename)
        files.append(local_filename)

#     # read the listing of directories from the list of raw data years, these should all be 4-digit years
#     with ftplib.FTP() as ftp:
# 
#         ftp.connect('ftp://filsrv.cicsnc.org/')
#         ftp.login('anonymous')
#         ftp.cwd('olivier/data_CMORPH_NIDIS/02_RAW/%s/%s%s' % year, year, str(month).zfill(2))
#         ls = ftp.mlsd()
#     
#         files = []
#         for items in ls:
#             if item['type'] == 'file':
#                 local_filename = os.sep.join((destination_dir, item['name']))
#                 with open(local_filename, 'wb') as f:
#                     ftp.retrbinary('RETR %s' % filename, f.write)            
#                 files.append(local_filename)
# 
#         ftp.quit()
#         
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
    :param initial_month: the month within the initial year from which the day values should start, with 1: January, 2: February, etc.
    :param total_months: the total number of monthly increments (time steps measured in days) to be computed
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
def ingest_cmorph_to_netcdf_full(cmorph_dir,
                                 netcdf_file,
                                 data_descriptor_file_name='CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl',
                                 download_files=False,
                                 remove_files=False):
    """
    Ingests CMORPH daily precipitation files into a full period of record file containing monthly cumulative precipitation.
    
    :param cmorph_dir: work directory where CMORPH files are expected to be located, downloaded files will reside here
    :param netcdf_file: output NetCDF
    :param data_descriptor_file_name: file name of the data descriptor file in CMORPH directory
    :param download_files: if true then download data descriptor and data files from FTP, overwrites files in CMORPH work directory
    :param remove_files: if files were downloaded then remove them once operations have completed 
    """
    
    # download the descriptor file, if required
    descriptor_file = os.sep.join((cmorph_dir, data_descriptor_file_name))
    if download_files:
        _download_data_descriptor(descriptor_file)

    # read data description info
    data_desc = _read_description(descriptor_file)
    
    # clean up the file, if required
    if download_files and remove_files:
        os.remove(descriptor_file)
    
    # get the years covered
    years = _get_years()
        
    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lon', data_desc['xdef_count'])
        output_dataset.createDimension('lat', data_desc['ydef_count'])
    
        output_dataset.title = data_desc['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        x_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        y_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        
        # set the coordinate variables' attributes
        units_since_year = 1800
        time_variable.units = 'days since %s-01-01 00:00:00' % units_since_year
        x_variable.units = 'degrees east'
        y_variable.units = 'degrees north'
        
        # compute the time values 
        time_variable[:] = _compute_days(data_desc['start_date'].year,
                                         len(years) * 12,
                                         initial_month=data_desc['start_date'].month,
                                         units_start_year=units_since_year)
        
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
        lon_values = list(_frange(data_desc['xdef_start'], data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment']), data_desc['xdef_increment']))
        lat_values = list(_frange(data_desc['ydef_start'], data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment']), data_desc['ydef_increment']))
        x_variable[:] = np.array(lon_values, 'f4')
        y_variable[:] = np.array(lat_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data_variable = output_dataset.createVariable('prcp', 
                                                      'f8', 
                                                      ('time', 'lon', 'lat',), 
                                                      fill_value=data_desc['undef'])
        data_variable.units = 'mm'
        data_variable.standard_name = 'precipitation'
        data_variable.long_name = 'precipitation, monthly cumulative'
        data_variable.description = data_desc['title']

        # loop over each year/month, reading binary data from CMORPH files and adding into the NetCDF variable
        for year in years:
            for month in range(1, 13):

                # get the files for the month
                if download_files:
                    downloaded_files = _download_daily_files(cmorph_dir, year, month)
                                    
                # read all the data for the month as a sum from the daily values, assign into the appropriate slice of the variable
                data = _read_daily_cmorph_to_monthly_sum(cmorph_dir, data_desc, year, month)
                time_index = (year * 12) + month - 1
                data_variable[time_index, :, :] = np.reshape(data, (1, data_desc['xdef_count'], data_desc['ydef_count']))
        
                # clean up, if necessary
                if download_files and remove_files:
                    for file in downloaded_files:
                        os.remove(file)
                    
#-----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf_monthly(cmorph_dir, 
                                    descriptor_file,
                                    netcdf_file,
                                    year, 
                                    month,
                                    download_files,
                                    remove_files):
    """
    :param cmorph_dir:
    :param descriptor_file:
    :param netcdf_file:
    :param year:
    :param month:  
    :param download_files:    
    :param remove_files:    
    """
    
    # download (if necessary) and read data description info
    if download_files:
        _download_data_descriptor(descriptor_file)
    data_desc = _read_description(descriptor_file)

    # clean up    
    if download_files and remove_files:
        os.remove(descriptor_file)

    # download (if necessary) and read data description info
    if download_files:
        daily_files = _download_daily_files(cmorph_dir, year, month)

    # read and sum the daily values into an array for the month
    data = _read_daily_cmorph_to_monthly_sum(cmorph_dir, data_desc)
    
    # clean up, if necessary
    if download_files and remove_files:
        for file in daily_files:
            os.remove(file)

    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lon', data_desc['xdef_count'])
        output_dataset.createDimension('lat', data_desc['ydef_count'])
    
        output_dataset.title = data_desc['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        x_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        y_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        
        # set the coordinate variables' attributes
        units_start_year = 1800
        time_variable.units = 'days since %s-01-01 00:00:00' % units_start_year
        x_variable.units = 'degrees east'
        y_variable.units = 'degrees north'
        
        # compute the time value 
        start_date = datetime(units_start_year, 1, 1)
        file_date = datetime(year, month, 1)
        time_variable[:] = np.array([(file_date - start_date).days])
        
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
        lon_values = list(_frange(data_desc['xdef_start'], data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment']), data_desc['xdef_increment']))
        lat_values = list(_frange(data_desc['ydef_start'], data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment']), data_desc['ydef_increment']))
        x_variable[:] = np.array(lon_values, 'f4')
        y_variable[:] = np.array(lat_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data_variable = output_dataset.createVariable('prcp', 
                                                      data.dtype, 
                                                      ('time', 'lon', 'lat',), 
                                                      fill_value=data_desc['undef'])
        data_variable[:] = np.reshape(data, (1, data_desc['xdef_count'], data_desc['ydef_count']))
        
#-----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf_daily(cmorph_file, 
                                  descriptor_file,
                                  netcdf_file):
    
    # read data description info
    data_desc = _read_description(descriptor_file)
    
    # parse date fields from the file name
    data_date = datetime.strptime(cmorph_file[-8:], '%D%M%Y')  # example: "01011998"
    
    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lon', data_desc['xdef_count'])
        output_dataset.createDimension('lat', data_desc['ydef_count'])
    
        output_dataset.title = data_desc['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        x_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        y_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        
        # set the coordinate variables' attributes
        units_start_year = 1800
        time_variable.units = 'days since %s-01-01 00:00:00' % units_start_year
        x_variable.units = 'degrees east'
        y_variable.units = 'degrees north'
        
        # compute the time value 
        start_date = datetime(units_start_year, 1, 1)
        time_variable[:] = np.array([(data_date - start_date).days])
        
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
        lon_values = list(_frange(data_desc['xdef_start'], data_desc['xdef_start'] + (data_desc['xdef_count'] * data_desc['xdef_increment']), data_desc['xdef_increment']))
        lat_values = list(_frange(data_desc['ydef_start'], data_desc['ydef_start'] + (data_desc['ydef_count'] * data_desc['ydef_increment']), data_desc['ydef_increment']))
        x_variable[:] = np.array(lon_values, 'f4')
        y_variable[:] = np.array(lat_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data = np.fromfile(cmorph_file, 'f')
        if not data_desc['little_endian']:
            data = data.byteswap()
        data = np.ma.masked_values(data, data_desc['undef'])
        data_variable = output_dataset.createVariable('prcp', 
                                                      data.dtype, 
                                                      ('time', 'lon', 'lat',), 
                                                      fill_value=data_desc['undef'])
        data_variable[:] = np.reshape(data, (1, data_desc['xdef_count'], data_desc['ydef_count']))
        
#-----------------------------------------------------------------------------------------------------------------------
def _frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

#-----------------------------------------------------------------------------------------------------------------------
def _read_description(descriptor_file):
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

    return data_dict

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform ingest of binary CMORPH datasets to NetCDF.

    Example command line usage for reading daily values into a single daily NetCDF:
    
    $ python -u ingest_cmorph.py --in_file C:/home/data/cmorph_file.cm \
                                 --descriptor_file C:/home/data/cmorph_file.ctl \
                                 --out_file C:/home/data/cmorph_file.nc

    or for reading all daily files into a single NetCDF file with cumulative monthly precipitation for a single month:
    
    $ python -u ingest_cmorph.py --cmorph_dir C:/home/data/cmorph/raw \
                                 --descriptor_file C:/home/data/cmorph_file.ctl \
                                 --out_file C:/home/data/cmorph_file.nc
                                 
    or for reading all daily files for all months into a single NetCDF file with cumulative monthly precipitation 
    for full period of record (all months), with all files downloaded from FTP:
    
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
        parser.add_argument("--in_file", 
                            help="Binary CMORPH data file", 
                            required=False)
        parser.add_argument("--cmorph_dir", 
                            help="Directory containing daily binary CMORPH data files for a single month", 
                            required=True)
        parser.add_argument("--descriptor_file", 
                            help="Data descriptor file corresponding to the input binary CMORPH data file",
                            default='CMORPH_V1.0_RAW_0.25deg-DLY_00Z.ctl',
                            required=False)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables read from the input data", 
                            required=True)
        parser.add_argument("--download_files", 
                            help="Download data from FTP, saving files in the CMORPH data directory specified by --cmorph_dir",
                            type=bool,
                            default=False, 
                            required=False)
        parser.add_argument("--remove_files", 
                            help="Remove downloaded data files from the CMORPH data directory if specified by --download_files",
                            type=bool,
                            default=False, 
                            required=False)
        args = parser.parse_args()

        # perform the ingest to NetCDF
#         ingest_cmorph_to_netcdf_daily(args.in_file, args.descriptor_file, args.out_file)
#         ingest_cmorph_to_netcdf_monthly(args.cmorph_dir, args.descriptor_file, args.out_file, year, month, args.download_files, args.remove_files)
        ingest_cmorph_to_netcdf_full(args.cmorph_dir,
                                     args.out_file,
                                     data_descriptor_file_name=args.descriptor_file,
                                     download_files=args.download_files,
                                     remove_files=args.remove_files)
        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    