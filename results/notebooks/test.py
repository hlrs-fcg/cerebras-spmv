df_small =  pd.read_csv("../../cluster/results/data/2026-01-20_strong-scaling-small.csv")
df_medium = pd.read_csv("../../cluster/results/data/2026-01-20_strong-scaling-medium.csv")
df_large =  pd.read_csv("../../cluster/results/data/2026-01-20_strong-scaling-large.csv")

df_small['problem'] = 'Small (500K NNZ)'
df_medium['problem'] = 'Medium (10M NNZ)'
df_large['problem'] = 'Large (40M NNZ)'

# Calculate both efficiency metrics
for df_temp in [df_small, df_medium, df_large]:
    # Compute-only efficiency (algorithmic)
    base_time_compute = df_temp.iloc[0]['compute_time_hw_avg']
    df_temp['speedup_compute'] = base_time_compute / df_temp['compute_time_hw_avg']
    df_temp['efficiency_compute'] = df_temp['speedup_compute'] / (df_temp['num_pes'] / df_temp['num_pes'].min()) * 100
    
    # Total iteration efficiency (practical)
    df_temp['total_iteration_time'] = (
        df_temp['compute_time_hw_avg'] + 
        df_temp['memcpy_h2d_time'] + 
        df_temp['memcpy_d2h_time'] +
        df_temp['prepare_matched_pairs_time'] +
        df_temp['gather_results_time']
    )
    base_time_total = df_temp.iloc[0]['total_iteration_time']
    df_temp['speedup_total'] = base_time_total / df_temp['total_iteration_time']
    df_temp['efficiency_total'] = df_temp['speedup_total'] / (df_temp['num_pes'] / df_temp['num_pes'].min()) * 100

# Single merged plot
fig, ax = plt.subplots(1, 1, figsize=(22, 9))

colors = {'small': 'C0', 'medium': 'C1', 'large': 'C2'}  # blue, orange, green

# Plot compute-only efficiency with circles and solid lines
for df_temp, label, color_key in [(df_small, 'Small', 'small'),
                                   (df_medium, 'Medium', 'medium'),
                                   (df_large, 'Large', 'large')]:
    ax.plot(df_temp['num_pes'], df_temp['efficiency_compute'], 
            marker='o', color=colors[color_key], linestyle='-',
            linewidth=2, markersize=12)

# Plot total iteration efficiency with squares and dashed lines
for df_temp, label, color_key in [(df_small, 'Small', 'small'),
                                   (df_medium, 'Medium', 'medium'),
                                   (df_large, 'Large', 'large')]:
    ax.plot(df_temp['num_pes'], df_temp['efficiency_total'], 
            marker='s', color=colors[color_key], linestyle='--',
            linewidth=2, markersize=12)

ax.set_ylim(0, 105)
ax.set_xlabel('Number of PEs')#, labelpad=15)
ax.set_ylabel('Parallel Efficiency (%)') labelpad=15)
ax.set_xscale('log', base=2)

all_pes = sorted(set(df_small['num_pes'].tolist() + 
                     df_medium['num_pes'].tolist() + 
                     df_large['num_pes'].tolist()))
#ax.set_xticks(all_pes)
#ax.set_xticklabels([f'{int(pe):,}' for pe in all_pes], rotation=45, ha='right')

extended_ticks = [2**i for i in range(8, 23)]  # 2^14 to 2^18
ax.set_xticks(extended_ticks)
ax.set_xticklabels([f'{int(pe):,}' for pe in extended_ticks], rotation=45, ha='right')



ax_top = ax.secondary_xaxis('top')
ax_top.set_xticks(extended_ticks)
ax_top.set_xticklabels([f'$2^{{{int(np.log2(pe))}}}$' for pe in extended_ticks])
ax_top.minorticks_off()

# Custom legend with combined entries
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
legend_elements = []
for label, color_key in [('Small', 'small'), ('Medium', 'medium'), ('Large', 'large')]:
    # Create a tuple of two line handles to show both styles
    solid_line = Line2D([0], [0], marker='o', color=colors[color_key], linestyle='-', 
                        markersize=20, linewidth=2)
    dashed_line = Line2D([0], [0], marker='s', color=colors[color_key], linestyle='--', 
                         markersize=12, linewidth=2)
    legend_elements.append((solid_line, dashed_line))
    
legend_labels = ['Small', 'Medium', 'Large']
ax.set_xlim(None,4494304)

ax.legend([(e[0], e[1]) for e in legend_elements], legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='center right')#, bbox_to_anchor=(1.02, 0.5, 0.20, 0.0))
            #mode='expand', borderaxespad=0)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/plots/strong_scaling-efficiencies.png', bbox_inches='tight')
