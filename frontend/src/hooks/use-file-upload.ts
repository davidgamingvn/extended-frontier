"use client";

import { useState } from "react";
import { BACKEND_URL } from "@/lib/utils";

const useFloorPlanUpload = () => {
  const [floorPlanUrl, setFloorPlanUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      // Upload the USDZ file
      const uploadResponse = await fetch(`${BACKEND_URL}/upload_usdz`, {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload USDZ file");
      }

      // Retrieve the floor plan image
      const timestamp = new Date().getTime();
      const floorPlanResponse = await fetch(
        `${BACKEND_URL}/get_floor_plan?t=${timestamp}`
      );

      if (!floorPlanResponse.ok) {
        throw new Error("Failed to fetch floor plan image");
      }

      const imageBlob = await floorPlanResponse.blob();
      const imageObjectUrl = URL.createObjectURL(imageBlob);
      setFloorPlanUrl(imageObjectUrl);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setIsUploading(false);
    }
  };

  return { floorPlanUrl, isUploading, uploadFile };
};

export default useFloorPlanUpload;
