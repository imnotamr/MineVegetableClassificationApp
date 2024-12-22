# Keep track of prediction history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Create a section for uploading images
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    # Multiple file uploader widget
    uploaded_files = st.file_uploader(
        "Choose vegetable images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display uploaded image
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True, output_format="auto")

            # Class Labels
            class_labels = [
                'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
            ]

            # Preprocess the uploaded image
            try:
                img = load_img(uploaded_file, target_size=(150, 150))  # Resize to match the input shape of the model
                img_array = img_to_array(img) / 255.0  # Normalize the image
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Make a prediction
                predictions = model.predict(img_array)
                predicted_class = class_labels[np.argmax(predictions)]

                # Add prediction to history
                st.session_state["history"].append({"Image": uploaded_file.name, "Prediction": predicted_class})

                # Display the predicted class
                st.markdown(f'**Predicted Vegetable:** {predicted_class}')

                # Display confidence scores as a horizontal bar chart
                confidence_df = pd.DataFrame({
                    'Class': class_labels,
                    'Confidence': predictions[0]
                }).sort_values(by='Confidence', ascending=True)

                fig = px.bar(
                    confidence_df,
                    x='Confidence',
                    y='Class',
                    orientation='h',
                    title="Confidence Scores for Each Class",
                    labels={'Confidence': 'Confidence (%)', 'Class': 'Vegetable Class'},
                    text=confidence_df['Confidence'].apply(lambda x: f'{x:.2%}')
                )
                fig.update_traces(marker_color='#2E86C1', textposition='outside')
                fig.update_layout(
                    xaxis_title="Confidence (%)",
                    yaxis_title="Vegetable Class",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")

    else:
        st.markdown('<div class="subtitle">Please upload an image to start classifying.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# View Prediction History
if st.button("View Prediction History"):
    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        st.write(history_df)
    else:
        st.info("No predictions have been made yet.")
