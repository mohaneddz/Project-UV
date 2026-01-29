from pathlib import Path
import pandas as pd
import sys


def find_date_column(df: pd.DataFrame):
	cols = df.columns
	# common date column name hints
	hints = ['date', 'day', 'time', 'datetime']
	for c in cols:
		low = c.lower()
		if any(h in low for h in hints):
			return c
	# fallback: try parsing each column and pick the one with fewest NaT
	best = None
	best_non_na = -1
	for c in cols:
		try:
			parsed = pd.to_datetime(df[c], errors='coerce')
			non_na = parsed.notna().sum()
			if non_na > best_non_na:
				best_non_na = non_na
				best = c
		except Exception:
			continue
	return best


def find_uv_column(df: pd.DataFrame, date_col: str = None):
	cols = df.columns
	# prefer columns with 'uv' in name
	for c in cols:
		if 'uv' in c.lower():
			return c
	# otherwise prefer 'index' combined with uv hints
	for c in cols:
		low = c.lower()
		if 'index' in low and ('uv' in low or 'uv' in ''.join(cols).lower()):
			return c
	# otherwise choose first numeric column that is not the date
	numeric = df.select_dtypes(include='number').columns.tolist()
	if date_col in numeric:
		numeric = [c for c in numeric if c != date_col]
	if numeric:
		return numeric[0]
	# last resort: return second column if first is date
	if len(cols) >= 2:
		return cols[1] if cols[0] == date_col else cols[0]
	return None


def compute_yearly_average(input_csv: Path, output_csv: Path):
	# read CSV, treat empty strings as NaN
	df = pd.read_csv(input_csv, na_values=['', 'NA', 'N/A', 'nan'])

	date_col = find_date_column(df)
	if date_col is None:
		raise ValueError('Could not find a date column in the CSV')

	df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
	if df['_parsed_date'].isna().all():
		raise ValueError(f'Failed to parse any dates from column "{date_col}"')

	df['year'] = df['_parsed_date'].dt.year

	# identify UV-related columns (case-insensitive)
	cols_lower = {c.lower(): c for c in df.columns}
	uv_index_col = None
	uva_col = None
	uvb_col = None
	for k, orig in cols_lower.items():
		if 'uv_index' in k or ('uv' in k and 'index' in k):
			uv_index_col = orig
		if k.endswith('uva') or 'allsky_sfc_uva' in k or (k == 'uva'):
			uva_col = orig
		if k.endswith('uvb') or 'allsky_sfc_uvb' in k or (k == 'uvb'):
			uvb_col = orig

	# coerce to numeric when columns exist
	if uv_index_col is not None:
		df[uv_index_col] = pd.to_numeric(df[uv_index_col], errors='coerce')
	if uva_col is not None:
		df[uva_col] = pd.to_numeric(df[uva_col], errors='coerce')
	if uvb_col is not None:
		df[uvb_col] = pd.to_numeric(df[uvb_col], errors='coerce')

	# make a combined UVA+UVB proxy column when available
	df['_uva_uvb_sum'] = None
	if uva_col and uvb_col:
		df['_uva_uvb_sum'] = df[uva_col].fillna(0) + df[uvb_col].fillna(0)
		# set to NaN where both were NaN
		both_na = df[uva_col].isna() & df[uvb_col].isna()
		df.loc[both_na, '_uva_uvb_sum'] = pd.NA
	elif uva_col:
		df['_uva_uvb_sum'] = df[uva_col]
	elif uvb_col:
		df['_uva_uvb_sum'] = df[uvb_col]

	# try to infer a scaling factor between UV_INDEX and UVA+UVB when overlap exists
	scale = None
	if uv_index_col is not None and '_uva_uvb_sum' in df.columns:
		mask = df[uv_index_col].notna() & df['_uva_uvb_sum'].notna() & (df['_uva_uvb_sum'] > 0)
		if mask.sum() >= 3:
			ratios = df.loc[mask, uv_index_col] / df.loc[mask, '_uva_uvb_sum']
			med = ratios.median()
			if pd.notna(med) and med > 0 and med < 1e6:
				scale = float(med)

	# additionally, try to use surface shortwave downwelling as a proxy when UV columns absent
	sw_col = None
	for k, orig in cols_lower.items():
		if 'allsky_sfc_sw_dwn' in k or 'sfc_sw_dwn' in k or 'sw_dwn' in k or 'sfc_sw' in k:
			sw_col = orig
			break
	if sw_col is not None:
		df[sw_col] = pd.to_numeric(df[sw_col], errors='coerce')

	sw_scale = None
	if uv_index_col is not None and sw_col is not None:
		mask2 = df[uv_index_col].notna() & df[sw_col].notna() & (df[sw_col] > 0)
		if mask2.sum() >= 3:
			ratios2 = df.loc[mask2, uv_index_col] / df.loc[mask2, sw_col]
			med2 = ratios2.median()
			if pd.notna(med2) and med2 > 0 and med2 < 1e6:
				sw_scale = float(med2)

	# build a single UV value column to aggregate:
	# prefer UV index; otherwise use scaled (UVA+UVB) proxy when possible
	def compute_uv_value(row):
		# use UV index if present
		if uv_index_col and pd.notna(row.get(uv_index_col)):
			return row.get(uv_index_col)
		# otherwise try UVA+UVB proxy
		val = row.get('_uva_uvb_sum')
		if pd.notna(val):
			if scale:
				return val * scale
			return val
		# otherwise try shortwave proxy
		if sw_col and pd.notna(row.get(sw_col)):
			if sw_scale:
				return row.get(sw_col) * sw_scale
			return row.get(sw_col)
		return pd.NA

	df['uv_value'] = df.apply(compute_uv_value, axis=1)

	# aggregate by year with counts
	agg = df.groupby('year').agg(
		uv_avg=('uv_value', 'mean'),
		valid_days=('uv_value', lambda s: int(s.notna().sum())),
		total_days=('year', 'size')
	).reset_index()
	agg['percent_valid'] = (agg['valid_days'] / agg['total_days'] * 100).round(2)

	# print to console: show only years with any valid data first, then others
	pd.set_option('display.max_rows', None)
	print(agg.to_string(index=False))

	# save to CSV
	agg.to_csv(output_csv, index=False)
	print(f"Saved yearly UV summary to {output_csv}")

	# Plot yearly averages
	try:
		import matplotlib.pyplot as plt
	except Exception:
		print('matplotlib not available; skipping plot. To enable plotting: pip install matplotlib', file=sys.stderr)
		return

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(agg['year'], agg['uv_avg'], marker='o', linestyle='-')
	ax.set_xlabel('Year')
	ax.set_ylabel('UV (avg)')
	ax.set_title('Algeria Yearly Average UV')
	# annotate percent_valid as text above points when less than 100%
	for _, r in agg.iterrows():
		if r['percent_valid'] < 100:
			ax.annotate(f"{r['percent_valid']}%", (r['year'], r['uv_avg'] if pd.notna(r['uv_avg']) else 0),
						textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)

	png_path = output_csv.with_suffix('.png')
	fig.tight_layout()
	fig.savefig(png_path)
	print(f"Saved plot to {png_path}")


def main():
	base = Path(__file__).parent
	input_csv = base / 'data' / 'UV-ALGERIA.csv'
	output_csv = base / 'Algeria-Yearly.csv'

	if not input_csv.exists():
		print(f'Input file not found: {input_csv}', file=sys.stderr)
		sys.exit(1)

	compute_yearly_average(input_csv, output_csv)


if __name__ == '__main__':
	main()

