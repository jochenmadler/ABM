# imports
library(tsbox)
library(xts)
library(tempdisagg)
library(tibble)
library(comprehenr)
options(max.print = 200)

# set up wd
path = 'C:\\Users\\joche\\FIM Kernkompetenzzentrum\\Paper Agent-based Modeling - Dokumente\\General\\01 Data\\PVGIS'
setwd(path)

# create hourly ts
hourly_generation = read.csv('hourly_radiation.csv')
time_index_h = seq(from = as.POSIXct("2019-01-01 00:00:00"), 
                   to = as.POSIXct("2019-12-31 23:00:00"), by = "day")

# define mock data
x_yearly = ts_tbl(ts(c(1076), start = 2019, frequency = 1))
x_monthly = ts_tbl(ts(sample(0:1000, size = 12), start = 2019, frequency = 12))
x_daily = ts_tbl(ts(sample(0:1000, size = 365, replace = T), start = 2019, frequency = 365))
x_hourly = ts_tbl(ts(sample(0:100, size = 8760, replace = T), start = 2019, frequency = 8760))

# redefine date format for all series
x_yearly$time = as.POSIXct(x_yearly$time)
x_monthly$time = as.POSIXct(x_monthly$time)
x_daily$time = as.POSIXct(x_daily$time)
x_hourly$time = as.POSIXct(x_hourly$time)

# yearly to monthly
m_y_m = td(x_yearly ~ 0 + x_monthly, method = 'denton-cholette', conversion = 'sum')
# -> not possible to disaggregate a single value
