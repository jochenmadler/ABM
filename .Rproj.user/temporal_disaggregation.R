# set up path and read in file
path = 'C:\\Users\\joche\\FIM Kernkompetenzzentrum\\Paper Agent-based Modeling - Dokumente\\General\\01 Data\\PVGIS'
#df_in = read.csv(file = filepath)
setwd(path)

# Tetraeder.solar yearly total radiation in Wh per square meter
global_radiation = 1076 * 1000
# Yearly timeseries of one year
annual_radiation = ts(global_radiation, start = c(2019), frequency=1)

# PVGIS hourly PV generation
h = read.csv('hourly_radiation.csv')
# Hourly timeseries of one year
first_hour = 24*(as.Date("2006-12-17 00:00:00")-as.Date("2006-1-1 00:00:00"))
fh = as.Date('2019-01-01 00:00:00')
hourly_generation = ts(data = h[2], start = c(2019,fh), frequency = 365*24)

# model
m1 = td(annual_radiation ~ 0 + hourly_generation, method = 'denton-cholette')
