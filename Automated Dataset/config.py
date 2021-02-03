clearCommand = ["--folder", "/store_00020001/DCIM/100CANON", "--delete-all-files", "-R"]
triggerCommand = ["--trigger-capture"]
downloadCommand = ["--get-all-files"]

# raw_and_large = ["--set-config", "imageformatcf=7"]
raw_and_large = ["--set-config", "imageformatsd=7"]
# small = ["--set-config", "imageformatcf=6"]
small = ["--set-config", "imageformatsd=4"]

short_exp = ["--set-config", "shutterspeed=52"]  # 1 / 4000
def_exp = ["--set-config", "shutterspeed=32"]  # 1 / 40
long_exp = ["--set-config", "shutterspeed=2"]  # 2

iso_init = ["--set-config", "iso=5"]
