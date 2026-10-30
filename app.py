import io
import json
import math
import time

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps


# Page config and UI texts in English
st.set_page_config(page_title="AI Receipt & Invoice Processor", page_icon="🧾")
st.title("🧾 AI Receipt & Invoice Processor")
st.write("Upload receipts or invoices (Images or PDFs) and download the extracted data as a structured Excel file.")

# File uploader - supports PDFs too
uploaded_files = st.file_uploader(
    "Select image(s) or PDF(s)...",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"Files selected: **{len(uploaded_files)}**")

    if st.button("Process and Generate Excel 🚀"):
        try:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        except Exception as e:
            st.error(
                "Gemini could not be initialized. Check that GEMINI_API_KEY is set "
                f"in Streamlit secrets. Details: {e}"
            )
            st.stop()

        all_receipts_data = []

        prompt = """
        Analyze this Romanian receipt or invoice. Extract exactly these six fields and
        return one valid JSON object:
        - "furnizor": supplier's name, not the customer or payment processor; string or null.
        - "cui": supplier's Romanian CUI/CIF as a string, or null if unavailable.
        - "data": issue date in YYYY-MM-DD format, or null if missing or ambiguous.
        - "total_brut": final gross transaction total including VAT as a JSON number,
          or null if uncertain. Do not confuse it with the net subtotal, amount tendered, or change.
        - "cota_tva": explicit VAT rate as a percentage number (e.g. 19, not 0.19).
          If multiple rates exist, return the highest; use null if unknown. Do not assume 0%.
        - "categorie": choose one: Fuel, Food, Hardware, Services, Utilities, Transport,
          Accommodation, Office supplies, Telecommunications, Other; use null if unclear.

        Extract only information supported by the document. Never invent or estimate values.
        Use JSON null for missing or uncertain information, not zero or "N/A". Return
        numeric fields as JSON numbers, not strings. Return only raw JSON, without Markdown
        or explanations.
        """

        with st.spinner("AI is processing the documents..."):
            for file in uploaded_files:
                if file.size > 10 * 1024 * 1024:
                    st.error(f"Could not process '{file.name}': file exceeds the 10 MB limit.")
                    continue

                try:
                    # Determine payload format based on file extension
                    if file.name.lower().endswith(".pdf"):
                        payload_data = types.Part.from_bytes(
                            data=file.getvalue(), mime_type="application/pdf"
                        )
                    else:
                        image = ImageOps.exif_transpose(Image.open(file))
                        image.thumbnail((1500, 1500))
                        payload_data = image

                    max_retries = 4
                    response = None

                    # Exponential backoff for temporary service errors
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model="gemini-3.6-flash",
                                contents=[payload_data, prompt],
                            )
                            break
                        except Exception as api_err:
                            if "503" in str(api_err) and attempt < max_retries - 1:
                                time.sleep((attempt + 1) * 3)
                            else:
                                raise RuntimeError(f"Gemini API request failed: {api_err}") from api_err

                    if response is None or not response.text:
                        raise ValueError("Gemini returned an empty response.")

                    clean_json_text = response.text.strip()
                    if clean_json_text.startswith("```"):
                        lines = clean_json_text.splitlines()
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        clean_json_text = "\n".join(lines).strip()

                    try:
                        data = json.loads(clean_json_text)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Gemini returned invalid JSON (line {e.lineno}, column {e.colno})."
                        ) from e

                    if not isinstance(data, dict):
                        raise ValueError("Gemini's response must be a JSON object.")

                    expected_fields = [
                        "furnizor", "cui", "data", "total_brut", "cota_tva", "categorie"
                    ]
                    # Missing keys are represented as unknown values, never as zero.
                    data = {field: data.get(field) for field in expected_fields}

                    # Normalize blank and common unknown-value placeholders to null.
                    unknown_text = {"", "n/a", "na", "unknown", "null", "none", "-"}
                    for field in expected_fields:
                        value = data[field]
                        if isinstance(value, str) and value.strip().lower() in unknown_text:
                            data[field] = None

                    for field in ("furnizor", "cui", "data", "categorie"):
                        value = data[field]
                        if value is not None and not isinstance(value, str):
                            raise ValueError(f"Field '{field}' must be text or null.")

                    for field in ("total_brut", "cota_tva"):
                        value = data[field]
                        if value is not None:
                            if isinstance(value, bool) or not isinstance(value, (int, float)):
                                raise ValueError(f"Field '{field}' must be a number or null.")
                            if not math.isfinite(value):
                                raise ValueError(f"Field '{field}' must be a finite number or null.")

                    data["filename"] = file.name
                    all_receipts_data.append(data)

                except Exception as e:
                    st.error(
                        f"Could not process '{file.name}': {e} "
                        "Check that the file is a valid image or PDF and try again."
                    )

                # Pause between files to avoid hitting the RPM rate limit
                time.sleep(3)

        if all_receipts_data:
            st.success(
                f"Finished processing: {len(all_receipts_data)} of {len(uploaded_files)} "
                "document(s) processed successfully."
            )

            # Create DataFrame with stable columns, including null values
            column_order = [
                "filename", "furnizor", "cui", "data", "total_brut", "cota_tva", "categorie"
            ]
            df = pd.DataFrame(all_receipts_data, columns=column_order)
            st.dataframe(df, use_container_width=True)

            # Generate Excel in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Excel (.xlsx)",
                data=excel_buffer,
                file_name="processed_documents.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("No documents were processed successfully. Review the errors above and try again.")
