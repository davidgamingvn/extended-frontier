"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/hooks/use-toast";
import { BACKEND_URL } from "@/lib/utils";
import { Router } from "lucide-react";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

interface Point {
  id: number;
  x: number;
  y: number;
}

interface ImageDimensions {
  width: number;
  height: number;
  top: number;
  left: number;
}

export default function BlankMap() {
  const [point, setPoint] = useState<Point | null>(null);
  const [imageDimensions, setImageDimensions] =
    useState<ImageDimensions | null>(null);
  const [imageUrl, setImageUrl] = useState("/latest_floor_plan.png");
  const [isLoading, setIsLoading] = useState(false);
  const [extendersCount, setExtendersCount] = useState(2);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const updateImageDimensions = () => {
      if (imageRef.current && imageContainerRef.current) {
        const containerRect = imageContainerRef.current.getBoundingClientRect();
        const imageRect = imageRef.current.getBoundingClientRect();

        setImageDimensions({
          width: imageRect.width,
          height: imageRect.height,
          top: imageRect.top - containerRect.top,
          left: imageRect.left - containerRect.left,
        });
      }
    };

    const img = imageRef.current;
    if (img) {
      if (img.complete) {
        updateImageDimensions();
      } else {
        img.onload = updateImageDimensions;
      }
    }

    const resizeObserver = new ResizeObserver(updateImageDimensions);
    if (imageContainerRef.current) {
      resizeObserver.observe(imageContainerRef.current);
    }

    window.addEventListener("resize", updateImageDimensions);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateImageDimensions);
    };
  }, [imageUrl]);

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageDimensions || isLoading) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (
      x < imageDimensions.left ||
      x > imageDimensions.left + imageDimensions.width ||
      y < imageDimensions.top ||
      y > imageDimensions.top + imageDimensions.height
    ) {
      return;
    }

    const relativeX = x - imageDimensions.left;
    const relativeY = y - imageDimensions.top;

    setPoint({
      id: Date.now(),
      x: relativeX,
      y: relativeY,
    });
  };

  const handleRemovePoint = () => {
    setPoint(null);
  };

  const handleDone = async () => {
    if (!point) {
      toast({
        title: "No router added",
        description: "Please add a router to the map before proceeding.",
        variant: "destructive",
      });
      return;
    }

    if (!fileInputRef.current?.files?.[0]) {
      toast({
        title: "No floor plan uploaded",
        description: "Please upload a floor plan image before proceeding.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsLoading(true);

      const formData = new FormData();
      formData.append("file", fileInputRef.current.files[0]);

      if (!imageDimensions) {
        toast({
          title: "Error",
          description: "Image dimensions are not available.",
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }

      formData.append("router_x", (point.x / imageDimensions.width).toString());
      formData.append(
        "router_y",
        (point.y / imageDimensions.height).toString()
      );
      formData.append("n_extenders", extendersCount.toString());

      // Save the point
      const saveResponse = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!saveResponse.ok) {
        throw new Error("Failed to save map");
      }

      // Get the updated image
      const data = await saveResponse.json();
      console.log(data);

      let timestamp = new Date().getTime();
      const imageResponse = await fetch(
        `${BACKEND_URL}/get_image?t=${timestamp}`
      );
      if (!imageResponse.ok) {
        throw new Error("Failed to fetch updated image");
      }
      const imageBlob = await imageResponse.blob();
      const imageObjectUrl = URL.createObjectURL(imageBlob);

      // Update the image with a cache-busting parameter
      setImageUrl(imageObjectUrl);
      setPoint(null); // Clear the point since it's now part of the image

      toast({
        title: "Success",
        description: "Map updated successfully!",
      });
    } catch (error) {
      console.error("Error updating map:", error);
      toast({
        title: "Error",
        description: "Failed to update map. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-lg font-medium">
            Map the location of your router in your house
          </h1>
          <Button
            onClick={handleDone}
            variant="default"
            className="bg-red-600 hover:bg-red-700 text-white"
            disabled={isLoading}
          >
            {isLoading ? "Updating..." : "Done"}
          </Button>
        </div>
      </div>

      <div className="container mx-auto p-4 flex flex-col md:flex-row gap-8">
        <div className="md:w-1/4 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-red-600">
                Add Router
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Click on the floor plan to add your router. You can only add one
                router to the map.
              </p>
              <div className="flex items-center space-x-2 mt-4">
                <Router className="w-5 h-5 text-red-600" />
                <span>Router</span>
              </div>
            </CardContent>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-red-600">
                How many extenders do you need?
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full max-w-xs">
                <Label
                  htmlFor="extenders-count"
                  className="text-sm font-medium text-gray-700"
                >
                  Number of Extenders
                </Label>
                <Input
                  id="extenders-count"
                  type="number"
                  min={0}
                  value={extendersCount}
                  onChange={(e) =>
                    setExtendersCount(parseInt(e.target.value) || 0)
                  }
                  className="mt-1"
                />
              </div>
              <div className="w-full max-w-xs mt-4">
                <Label
                  htmlFor="file-upload"
                  className="text-sm font-medium text-gray-700"
                >
                  Upload Floor Plan
                </Label>
                <Input
                  id="file-upload"
                  type="file"
                  ref={fileInputRef}
                  className="mt-1"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="md:w-3/4">
          <div
            ref={imageContainerRef}
            className={`relative w-100 h-[calc(100vh-12rem)] bg-muted rounded-lg shadow-inner overflow-hidden ${
              isLoading ? "cursor-wait" : "cursor-crosshair"
            } border`}
            onClick={handleImageClick}
          >
            <div className="relative w-full h-full flex items-center justify-center">
              <div ref={imageRef} className="relative">
                <Image
                  src={imageUrl}
                  alt="Floor Plan"
                  className="pointer-events-none max-h-full max-w-full object-contain"
                  width={400}
                  height={400}
                  priority
                />
                {imageDimensions && point && (
                  <div
                    key={point.id}
                    className="absolute w-10 h-10 -ml-4 -mt-4 rounded-full flex items-center justify-center cursor-pointer transition-all hover:scale-110 bg-red-600"
                    style={{ left: point.x, top: point.y }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemovePoint();
                    }}
                  >
                    <Router className="w-6 h-6 text-white" />
                  </div>
                )}
              </div>
            </div>
            <Card className="absolute bottom-4 right-4 p-2 text-sm">
              Click to add router • Click on the router to remove
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
