from WIND_bkg_rejection.utils import loss_analysis
import os

base_path = "/home/yujin/projects/wind/BKG_rejection/CNN/logs/test/version_0"
csv_path = os.path.join(base_path, "metrics.csv")
output_path = base_path
loss_analysis(csv_path=csv_path, output_path=base_path, png_title="loss_curve.png" )