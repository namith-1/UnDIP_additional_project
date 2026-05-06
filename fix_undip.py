with open(r'c:\Users\likhi\OneDrive\Documents\prev_files\UDIP\undip_train.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("np.save(os.path.join(dataset_out, 'A_est.npy'), A_est)", "np.save(os.path.join(dataset_out, 'A_est.npy'), A_est_matched)")
content = content.replace("np.save(os.path.join(dataset_out, 'A_est_avg.npy'), A_est_avg)", "np.save(os.path.join(dataset_out, 'A_est_avg.npy'), A_avg_matched)")

content = content.replace("print(f\"    Mean SAD (SiVM endmembers): {mean_sad_val:.4f}°\")", "print(f\"    Mean SAD (SiVM endmembers): {mean_sad_val:.4f} deg\")")
content = content.replace("print(f\"    RMSE (UnDIP final)        : {rmse_est:.4f}%\")", "print(f\"    RMSE (UnDIP final, best perm) : {rmse_est:.4f}%\")")
content = content.replace("print(f\"    RMSE (UnDIP avg)          : {rmse_avg:.4f}%\")", "print(f\"    RMSE (UnDIP avg, best perm)   : {rmse_avg:.4f}%\")")
content = content.replace("print(f\"  SADs (per endmember): {[f'{s:.2f}°' for s in sads]}\")", "print(f\"  SADs (per endmember): {[f'{s:.2f} deg' for s in sads]}\")")
content = content.replace("print(f\"  Mean SAD: {mean_sad_val:.4f}°\")", "print(f\"  Mean SAD: {mean_sad_val:.4f} deg\")")

with open(r'c:\Users\likhi\OneDrive\Documents\prev_files\UDIP\undip_train.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fix applied successfully.")
