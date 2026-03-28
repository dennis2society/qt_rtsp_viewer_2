from onvif import ONVIFCamera

#ip = "192.168.1.100"
ip = "192.168.1.104"
port = 8080
username = "wakko"
password = "d2srockz"

cam = ONVIFCamera(ip, port, username, password)

devicemgmt = cam.create_devicemgmt_service()
caps = devicemgmt.GetCapabilities({'Category': 'All'})

print(caps)

print("############ setting brightness ############")
media = cam.create_media_service()
imaging = cam.create_imaging_service()

video_sources = media.GetVideoSources()
token = video_sources[0].token

request = imaging.create_type('SetImagingSettings')
request.VideoSourceToken = token
request.ImagingSettings = {
    "Brightness": 60
}
request.ForcePersistence = True

imaging.SetImagingSettings(request)

print("Brightness set")

options = imaging.GetOptions({'VideoSourceToken': token})
print(options.Brightness)

settings = imaging.GetImagingSettings({'VideoSourceToken': token})
print(settings.Exposure)

settings = imaging.GetImagingSettings({'VideoSourceToken': token})
print("Current settings:", settings)

options = imaging.GetOptions({'VideoSourceToken': token})
print("Supported options:", options)
