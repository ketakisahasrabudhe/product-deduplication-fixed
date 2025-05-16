import streamlit as st
import pandas as pd
import io
from deduplication_utils import deduplicate_products

st.set_page_config(page_title="Product Deduplication", layout="wide")
st.title("Product Deduplication")

uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        # Reset index before displaying to avoid messy numbering
        st.dataframe(df.head().reset_index(drop=True), use_container_width=True)
        
        # Display column information
        st.subheader("Detected Columns")
        columns = df.columns.tolist()
        st.write(f"Found {len(columns)} columns: {', '.join(columns)}")
        
        # Check for important columns
        required_columns = ["title"]
        missing_required = [col for col in required_columns if col not in columns]
        
        if missing_required:
            st.warning(f"Missing recommended columns: {', '.join(missing_required)}. " 
                       f"The deduplication might be less accurate without title information.")
        
        # Check if we have price for lowest price strategy
        has_price = "price" in columns
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        st.stop()

    # Sidebar settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.99, 0.85, 0.01, 
                                 help="Higher values require products to be more similar to be considered duplicates")
    
    # Only show "Lowest Price" option if price column exists
    auto_pick_options = ["First Seen"]
    if "price" in df.columns:
        auto_pick_options.insert(0, "Lowest Price")
    
    auto_pick = st.sidebar.selectbox(
        "Pick representative by:", 
        auto_pick_options,
        help="How to choose which product to keep from a group of duplicates"
    )
    
    model_options = {
        "MiniLM": "all-MiniLM-L6-v2",
        "MPNet": "all-mpnet-base-v2",
        "Paraphrase-MiniLM": "paraphrase-MiniLM-L6-v2"
    }
    
    model_label = st.sidebar.selectbox(
        "Sentence Transformer Model", 
        list(model_options.keys()),
        help="Model used for calculating text similarity"
    )
    model_choice = model_options[model_label]
    
    # Optional settings expander
    with st.sidebar.expander("Advanced Settings"):
        # Display column mapping for flexibility
        column_mapping = {}
        default_columns = {
            "title": "title", 
            "brand": "brand", 
            "attributes": "attributes",
            "price": "price",
            "product_id": "product_id"
        }
        
        st.write("Column Mapping (leave blank if not available)")
        for field, default in default_columns.items():
            options = [""] + df.columns.tolist()
            default_idx = options.index(default) if default in options else 0
            column_mapping[field] = st.selectbox(f"{field} column", options, index=default_idx)
    
    if st.button("Deduplicate Products"):
        with st.spinner("Processing... Please wait..."):
            try:
                # Apply column mapping if in advanced settings
                if any(column_mapping.values()):
                    # Create a copy with renamed columns
                    mapped_df = df.copy()
                    
                    # Rename columns based on mapping
                    rename_dict = {v: k for k, v in column_mapping.items() if v}
                    if rename_dict:
                        mapped_df = mapped_df.rename(columns=rename_dict)
                    
                    # Use the mapped dataframe
                    df_to_process = mapped_df
                else:
                    df_to_process = df
                
                # Run deduplication
                df_deduped, pairs = deduplicate_products(df_to_process, threshold, model_choice, auto_pick)
                
                # Count duplicates removed
                dupes_removed = len(df) - len(df_deduped)
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        
        # Display results
        st.success(f"Done! Removed {dupes_removed} duplicate products ({dupes_removed/len(df)*100:.1f}%).")
        
        # Show duplicate pairs if any were found
        if not pairs.empty:
            with st.expander("View Duplicate Pairs"):
                st.dataframe(pairs.reset_index(drop=True), use_container_width=True)
        
        st.subheader("Deduplicated Data")
        # Reset index before displaying to avoid messy numbering
        st.dataframe(df_deduped.reset_index(drop=True), use_container_width=True)
        
        # Show how many rows are in the output
        st.info(f"Output contains {len(df_deduped)} rows from original {len(df)} rows.")
        
        # Provide download button
        output = io.BytesIO()
        df_deduped.to_excel(output, index=False, engine='openpyxl')
        st.download_button(
            "Download Deduplicated Excel", 
            data=output.getvalue(),
            file_name="deduplicated_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )