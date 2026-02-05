
import json
import os

notebook_path = r'c:\Users\PC\Documents\code\uv-index\Project-UV\health_assessment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- Update Cell 1: Markdown Title ---
title_found = False
for cell in cells:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "Section 1: Algeria UV Exposure Landscape" in source:
            cell['source'] = [
                "## 🌍 Section 1: Algeria UV Seasonality Analysis\n",
                "Analysis of temporal UV Index variations to identify high-risk periods throughout the year."
            ]
            title_found = True
            print("Updated Markdown Title.")
            break

if not title_found:
    print("Warning: Markdown Title cell not found.")

# --- Update Cell 2: Algeria Analysis Code ---
code_found = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "algeria_uv_by_admin_region.csv" in source:
            cell['source'] = [
                "# Load Algeria Time-Series Data\n",
                "try:\n",
                "    # Use the available time-series data\n",
                "    algeria_df = pd.read_csv('data/algeria/Algeria.csv')\n",
                "    \n",
                "    # Convert Date to datetime\n",
                "    algeria_df['Date'] = pd.to_datetime(algeria_df['Date'])\n",
                "    \n",
                "    # Extract Month\n",
                "    algeria_df['Month'] = algeria_df['Date'].dt.month\n",
                "    algeria_df['Month_Name'] = algeria_df['Date'].dt.month_name()\n",
                "    \n",
                "    # Calculate Monthly Means\n",
                "    monthly_uv = algeria_df.groupby('Month')['ALLSKY_SFC_UV_INDEX'].mean().reset_index()\n",
                "    # Map month numbers to names for plotting order\n",
                "    import calendar\n",
                "    monthly_uv['Month_Name'] = monthly_uv['Month'].apply(lambda x: calendar.month_name[x])\n",
                "    \n",
                "    # 🗺️ Visualization: Monthly UV Trends\n",
                "    plt.figure(figsize=(12, 7))\n",
                "    \n",
                "    # Color map from cool to hot\n",
                "    norm = plt.Normalize(monthly_uv['ALLSKY_SFC_UV_INDEX'].min(), monthly_uv['ALLSKY_SFC_UV_INDEX'].max())\n",
                "    colors = plt.cm.inferno(norm(monthly_uv['ALLSKY_SFC_UV_INDEX']))\n",
                "    \n",
                "    barplot = sns.barplot(x='Month_Name', y='ALLSKY_SFC_UV_INDEX', data=monthly_uv, palette='inferno')\n",
                "    \n",
                "    plt.title('📅 Algeria: Mean UV Index by Month (Seasonal Risk)', pad=20)\n",
                "    plt.xlabel('Month')\n",
                "    plt.ylabel('Mean UV Index')\n",
                "    plt.xticks(rotation=45)\n",
                "    \n",
                "    # Add value labels\n",
                "    for i, p in enumerate(barplot.patches):\n",
                "        height = p.get_height()\n",
                "        plt.text(p.get_x() + p.get_width()/2., height + 0.05, \n",
                "                 f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
                "                 \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "    \n",
                "    peak_month_row = monthly_uv.loc[monthly_uv['ALLSKY_SFC_UV_INDEX'].idxmax()]\n",
                "    peak_month = peak_month_row['Month_Name']\n",
                "    peak_val = peak_month_row['ALLSKY_SFC_UV_INDEX']\n",
                "    print(f\"⚠️ Peak Risk Month: {peak_month} (UV Index: {peak_val:.2f})\")\n",
                "    \n",
                "    # Prepare standard column for later summary if needed\n",
                "    algeria_df['uv_mean'] = algeria_df['ALLSKY_SFC_UV_INDEX']\n",
                "\n",
                "except Exception as e:\n",
                "    print(f\"⚠️ Error loading Algeria data: {e}\")"
            ]
            cell['outputs'] = [] # Clear outputs to avoid confusion before re-run
            code_found = True
            print("Updated Algeria Analysis Code.")
            break

if not code_found:
    print("Warning: Algeria Analysis Code cell not found.")

# --- Update Cell 3: Summary Statistics ---
summary_found = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "HEALTH ASSESSMENT SUMMARY" in source:
            # We need to replace the Algeria part of the summary
            old_algeria_summary = "print(f'   • High-risk southern regions: UV > 2.0')" # One distinctive line
            
            # Since the logic inside the print statements is what matters, I'll rewrite the whole cell source to be safe
            # utilizing the variables we know will exist ('algeria_df', 'peak_month' etc from previous cell likely available in context)
            # Actually, variables from previous cells are available if run in order.
            # But let's make it robust accessing the dataframe directly again or just generic stats.
            
            new_source = [
                "# Summary Statistics\n",
                "print('='*60)\n",
                "print('📊 HEALTH ASSESSMENT SUMMARY')\n",
                "print('='*60)\n",
                "\n",
                "print('\\n🌍 ALGERIA UV EXPOSURE:')\n",
                "if 'algeria_df' in locals():\n",
                "    print(f'   • Mean Annual UV Index: {algeria_df[\"ALLSKY_SFC_UV_INDEX\"].mean():.2f}')\n",
                "    print(f'   • Peak Season: {peak_month} (Index: {peak_val:.2f})')\n",
                "    print(f'   • Data Source: Time-series analysis (1981-2023)')\n",
                "else:\n",
                "    print('   • Data not loaded.')\n",
                "\n",
                "print('\\n🇬🇧 UK SUNBURN-UV CORRELATION:')\n",
                "if 'correlation' in locals():\n",
                "    print(f'   • Correlation (r): {correlation:.3f}')\n",
                "    print(f'   • Peak months: June-July')\n",
                "\n",
                "print('\\n🏥 NORDIC CANCER ANALYSIS:')\n",
                "if 'cancer_agg' in locals():\n",
                "    for _, row in cancer_agg.iterrows():\n",
                "        print(f'   • {row[\"Country\"]}: {row[\"Incidence_Per_100K\"]:.2f}/100K incidence')\n",
                "\n",
                "print('\\n✅ KEY FINDINGS:')\n",
                "print('   1. Strong seasonality in Algeria with intense summer UV peaks.')\n",
                "print('   2. Strong positive correlation between UV and acute sunburn in UK (r=~0.66).')\n",
                "print('   3. Nordic paradox confirmed: lower UV countries show higher melanoma rates.')\n",
                "print('   4. Public health focus: Summer protection in south, behavioral change in north.')"
            ]
            cell['source'] = new_source
            cell['outputs'] = []
            summary_found = True
            print("Updated Summary Statistics Code.")
            break

if not summary_found:
    print("Warning: Summary Statistics cell not found.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
