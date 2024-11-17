"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { toast } from "@/hooks/use-toast";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Router, Plug } from "lucide-react";

type PointType = "router" | "outlet";

interface Point {
  id: number;
  x: number;
  y: number;
  type: PointType;
}

interface ImageDimensions {
  width: number;
  height: number;
  top: number;
  left: number;
}

export default function BlankMap() {
  const [points, setPoints] = useState<Point[]>([]);
  const [selectedType, setSelectedType] = useState<PointType>("outlet");
  const [imageDimensions, setImageDimensions] =
    useState<ImageDimensions | null>(null);
  const [imageUrl, setImageUrl] = useState("/floorplan1.jpg");
  const [isLoading, setIsLoading] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);

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
  }, [imageUrl]); // Added imageUrl as dependency to recalculate dimensions when image changes

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

    if (selectedType === "router" && points.some((p) => p.type === "router")) {
      toast({
        title: "Cannot add router",
        description: "Only one router can be added to the map.",
        variant: "destructive",
      });
      return;
    }

    const relativeX = x - imageDimensions.left;
    const relativeY = y - imageDimensions.top;

    setPoints([
      ...points,
      {
        id: Date.now(),
        x: relativeX,
        y: relativeY,
        type: selectedType,
      },
    ]);
  };

  const handleRemovePoint = (id: number) => {
    setPoints(points.filter((point) => point.id !== id));
  };

  const handleDone = async () => {
    try {
      setIsLoading(true);

      // Save the points
      const saveResponse = await fetch("/api/save-map", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          points: points.map((point) => ({
            x: point.x,
            y: point.y,
            type: point.type,
          })),
          mapDimensions: imageDimensions,
        }),
      });

      if (!saveResponse.ok) {
        // throw new Error('Failed to save map')
      }

      // Get the updated image
      const imageResponse = await fetch("/api/get-map");
      if (!imageResponse.ok) {
        // throw new Error('Failed to get updated map')
      }

      const { imageUrl: newImageUrl } = await imageResponse.json();

      // Update the image with a cache-busting parameter
      setImageUrl(`${newImageUrl}?t=${Date.now()}`);
      setPoints([]); // Clear the points since they're now part of the image

      toast({
        title: "Success",
        description: "Map updated successfully!",
      });
    } catch (error) {
      //   console.error('Error updating map:', error)
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
            Map the location of your router and electrical outlets in
            your house
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
                Add Network Points
              </CardTitle>
            </CardHeader>
            <CardContent>
              <RadioGroup
                value={selectedType}
                onValueChange={(value: PointType) => setSelectedType(value)}
                className="space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="router"
                    id="router"
                    className="text-red-600"
                  />
                  <Label
                    htmlFor="router"
                    className="flex items-center space-x-2 cursor-pointer"
                  >
                    <div className="flex flex-row items-center gap-2">
                      <Router className="w-5 h-5" />
                      <span>Router (Max 1)</span>
                    </div>
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="outlet"
                    id="outlet"
                    className="text-red-600"
                  />
                  <Label
                    htmlFor="outlet"
                    className="flex items-center space-x-2 cursor-pointer"
                  >
                    <div className="flex flex-row items-center gap-2">
                      <Plug className="w-5 h-5" />
                      <span>Outlet</span>
                    </div>
                  </Label>
                </div>
              </RadioGroup>
            </CardContent>
          </Card>
        </div>

        <div className="md:w-3/4">
          <div
            ref={imageContainerRef}
            className={`relative w-full h-[calc(100vh-12rem)] bg-muted rounded-lg shadow-inner overflow-hidden ${
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
                  width={800}
                  height={600}
                  priority
                />
                {imageDimensions &&
                  points.map((point) => (
                    <div
                      key={point.id}
                      className={`absolute w-10 h-10 -ml-4 -mt-4 rounded-full flex items-center justify-center cursor-pointer transition-all hover:scale-110 ${
                        point.type === "router" ? "bg-red-600" : "bg-red-500"
                      }`}
                      style={{ left: point.x, top: point.y }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemovePoint(point.id);
                      }}
                    >
                      {point.type === "router" ? (
                        <Router className="w-6 h-6 text-white" />
                      ) : (
                        <Plug className="w-6 h-6 text-white" />
                      )}
                    </div>
                  ))}
              </div>
            </div>
            <Card className="absolute bottom-4 right-4 p-2 text-sm">
              Click to add points • Click on a point to remove
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
