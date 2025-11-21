import React, { useState } from "react";

const UploadForm = ({ onResult }) => {
  const [image, setImage] = useState(null);
  const [style, setStyle] = useState("mosaic");
  const [loading, setLoading] = useState(false);

  // ✅ Automatically pick backend URL depending on environment
const API_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "/api" : "https://style-generator-service-372485790810.asia-south1.run.app");


  // ✅ For debugging — see which API URL is being used
  console.log("Backend API URL:", API_URL);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      alert("Please upload an image!");
      return;
    }

    const formData = new FormData();
    formData.append("content_file", image);
    formData.append("style_name", style);
    formData.append("noise_octave", 4);
    formData.append("noise_seed", 1);

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/stylize_procedural/`, {
        method: "POST",
        body: formData,
        headers: {
          // ✅ CORS-safe headers — don’t set Content-Type manually for FormData
          Accept: "image/jpeg",
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Stylization failed! Server responded with ${response.status}: ${errorText}`
        );
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      onResult(imageUrl);
    } catch (err) {
      console.error("❌ Upload error:", err);
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <form onSubmit={handleSubmit} className="form">
        <h2>🎨 AI Style Transfer</h2>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          required
        />

        <select value={style} onChange={(e) => setStyle(e.target.value)}>
          <option value="mosaic">Mosaic</option>
          <option value="udnie">Udnie</option>
          <option value="rain-princess">Rain Princess</option>
          <option value="candy">Candy</option>
        </select>

        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Generate Style"}
        </button>
      </form>
    </div>
  );
};

export default UploadForm;
