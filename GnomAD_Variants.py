from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from matplotlib.colors import Normalize
import requests
from collections import defaultdict
import pandas as pd

app = Flask(__name__)


def fetch_ebi(gene_name,selected_features):
    """
    Map a canonical transcript ID to its corresponding UniProt accession number.
    """
    EBI_API_URL = "https://www.ebi.ac.uk/proteins/api/proteins?"

    params = {
        "offset": "0",
        "size": "100",  # Increase the size to get more results
        "reviewed": "true",
        "gene": gene_name,
        "isoform": "0",
        "organism": "Homo sapiens",
        "format": "json"
    }

    response = requests.get(EBI_API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    # Check for a perfect match in the name field
    perfect_match_entry = None
    for entry in data:
        if entry['id'] == gene_name+"_HUMAN":
            perfect_match_entry = entry
            break

    # If no perfect match is found, use the first entry
    if not perfect_match_entry:
        perfect_match_entry = data[0]

    sequence = perfect_match_entry['sequence']['sequence']
    features = [feature for feature in perfect_match_entry['features'] if feature['type'] in selected_features]
    length = perfect_match_entry['sequence']['length']
    df = pd.DataFrame([(feature['type'] + ': ' + feature['description'], feature['begin'], feature['end']) for feature in features], columns=['Name', 'Start', 'End'])

    return sequence, df, length

def fetch_missense_variants(gene_name):
    # GraphQL query to fetch missense variants for a specific gene
    query = f"""
    {{
      gene(gene_symbol: "{gene_name}", reference_genome: GRCh38) {{
        canonical_transcript_id
        variants(dataset: gnomad_r4) {{
          rsid
          consequence
          consequence_in_canonical_transcript
          gene_id
          gene_symbol
          transcript_id
          transcript_version
          hgvsc
          hgvsp
          lof
          lof_filter
          lof_flags
          hgvs
          exome {{
            ac
          }}
          genome {{
            ac
          }}
        }}
      }}
    }}
    """
    GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
    response = requests.post(GNOMAD_API_URL, json={'query': query})
    data = response.json()
    if 'gene' not in data['data'] or data['data']['gene'] is None:
        raise ValueError(f"No gene found with the name: {gene_name}")
    if 'data' not in data:
        print("Error: Unexpected API response.")
        print(data)  # Print the entire response to diagnose the issue
        return []

    # Filter for missense variants
    canonical_transcript = data['data']['gene']['canonical_transcript_id']
    missense_variants = [variant for variant in data['data']['gene']['variants'] if ((variant['consequence'] == 'missense_variant') | (variant['consequence'] == 'start_lost')) & (variant['transcript_id'] == data['data']['gene']['canonical_transcript_id'])]

    return missense_variants, canonical_transcript

def extract_aa_position(hgvsp):
    """Extract the amino acid position from the hgvsp string."""
    if not hgvsp:
        return None
    # Extract numbers from the hgvsp string (e.g., "p.Val600Glu" -> "600")
    position = ''.join([char for char in hgvsp if char.isdigit()])
    return int(position) if position else None

def plot_to_base64(gene_name,selected_features):
    # Your matplotlib code here using the gene_name
    MAX_WIDTH = 15
    aa_per_row = 100
    row_spacing = 2.0
    POP_SIZE = 807162

    # Fetch data
    variants, cannonical_transcript = fetch_missense_variants(gene_name)
    amino_acids, features, length = fetch_ebi(gene_name,selected_features)

    num_rows = int(np.ceil(len(amino_acids) / aa_per_row))

    # Calculate the actual width and height based on your data and constraints
    actual_width = min(MAX_WIDTH, aa_per_row * 0.5)
    actual_height = (num_rows * row_spacing + 1)*0.66

    fig, ax = plt.subplots(figsize=(actual_width, actual_height))
    ax.set_title("Frequency of Missense variants by AA Residue for " + gene_name)
    # Aggregate allele counts based on amino acid position
    aggregated_counts = defaultdict(int)

    for variant in variants:
        position = extract_aa_position(variant['hgvsp'])
        if position:
            exome_ac = variant['exome']['ac'] if variant['exome'] else 0
            genome_ac = variant['genome']['ac'] if variant['genome'] else 0
            aggregated_counts[position] += exome_ac + genome_ac

    # Convert the aggregated counts to a DataFrame
    df = pd.DataFrame(list(aggregated_counts.items()), columns=['Position', 'Allele_Count'])

    # Compute the frequency and add it as a new column
    df['Frequency'] = df['Allele_Count'] / (2 * POP_SIZE)

    # Return the DataFrame

    all_positions = pd.DataFrame({'Position': range(1, length + 1)})
    df = pd.merge(all_positions, df, on='Position', how='left').fillna(0)

    frequencies = df[['Frequency']].to_numpy()
    domains = [{"name": d["Name"], "start": int(d["Start"]), "end": int(d["End"])} for d in features.to_dict('records')]


    frequencies[frequencies <= 0] = 1e-10

    # Convert frequencies to -log10 values
    log_frequencies = np.log10(frequencies)

    # Compute the maximum -log10 value excluding the placeholder value
    max_log_frequency = min([val for val in log_frequencies if val != np.log10(1e-10)])

    # Create a color map based on the log frequencies using Normalize
    norm = Normalize(vmin=max_log_frequency, vmax=0)
    colors = plt.cm.viridis(norm(log_frequencies))

    # Set the color of any amino acid with the placeholder value to black
    colors[log_frequencies == np.log10(1e-10)] = [0, 0, 0, 1]

    # Number of amino acids per row

    # Calculate the number of rows needed
    num_rows = int(np.ceil(len(amino_acids) / aa_per_row))

    # Define the spacing between rows
    # # +1 for domain annotations space

    # Plot each amino acid as a rectangle
    for i, (aa, color) in enumerate(zip(amino_acids, colors)):
        row = i // aa_per_row
        col = i % aa_per_row
        y_coord = (num_rows - 1 - row) * row_spacing
        ax.add_patch(plt.Rectangle((col, y_coord), 1, 1, facecolor=color))
        ax.text(col + 0.5, y_coord + 0.5, aa, ha='center', va='center', color='w', fontsize=8)
        if col % 10 == 0:  # Tick mark for every 10th amino acid
            ax.text(col + 0.5, y_coord - 0.2, str(i+1), ha='center', va='center', fontsize=8)  # Position label

    # Plot domain annotations
    for domain in domains:
        start_row = domain["start"] // aa_per_row
        end_row = domain["end"] // aa_per_row
        start_col = (domain["start"] -1) % aa_per_row
        end_col = (domain["end"]) % aa_per_row

        for row in range(start_row, end_row + 1):
            y_coord = (num_rows - row) * row_spacing -1  # Adjusted y-coordinate to be closer
            if row == start_row:
                ax.add_patch(plt.Rectangle((start_col, y_coord), aa_per_row - start_col if row != end_row else end_col - start_col, 0.3, facecolor='gray', edgecolor='black'))  # Thinner rectangle
            elif row == end_row:
                ax.add_patch(plt.Rectangle((0, y_coord), end_col, 0.3, facecolor='gray', edgecolor='black'))  # Thinner rectangle
            else:
                ax.add_patch(plt.Rectangle((0, y_coord), aa_per_row, 0.3, facecolor='gray', edgecolor='black'))  # Thinner rectangle

            # Center the label within each rectangle
            if row == start_row:
                label_start = (start_col + aa_per_row) / 2 if row != end_row else (start_col + end_col) / 2
            elif row == end_row:
                label_start = end_col / 2
            else:
                label_start = aa_per_row / 2

            ax.text(label_start, y_coord + 0.15, domain["name"], ha='center', va='center', fontsize=6)

    ax.set_xlim(0, aa_per_row)
    ax.set_ylim(0, num_rows * row_spacing + 1)
    ax.axis('off')  # Turn off the axis

    # Display the colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), ax=ax, orientation='horizontal', pad=0.1, shrink=0.1, aspect=12)
    cbar.set_label('log10(Frequency)')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_base64 = None
    error_message = None

    feature_groups = {
        'functional_domains': ["DOMAIN", "REPEAT", "ZN_FING", "DNA_BIND"],
        'transmembrane_regions': ["TRANSMEM"],
        'active_binding_sites': ["ACT_SITE", "BINDING", "SITE"],
        'post_translational': ["MOD_RES", "LIPID", "CARBOHYD", "DISULFID", "CROSSLNK"]
    }

    selected_features = []
    if request.method == 'POST':
        gene_name = request.form['gene_name']
        selected_features = request.form.getlist('features')
        try:
            plot_base64 = plot_to_base64(gene_name, selected_features)
            plt.close()
        except ValueError as e:
            error_message = str(e)

    return render_template('index.html', plot_base64=plot_base64, feature_groups=feature_groups, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)