import { clsx, type ClassValue } from "clsx";
import { MapPin, Wifi, Zap } from "lucide-react";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const features = [
  {
    icon: MapPin,
    title: "Precise Locator",
    description:
      "Pinpoint the optimal location for your WiFi extender with our advanced algorithm.",
  },
  {
    icon: Zap,
    title: "Signal Strength Analyzer",
    description:
      "Real-time analysis of WiFi signal strength throughout your home.",
  },
  {
    icon: Wifi,
    title: "Visualization",
    description:
      "See a visual representation of your WiFi coverage before and after optimization.",
  },
];

export const steps = [
  {
    number: 1,
    title: "Download & Install",
    description:
      "Get the Frontier WiFi Companion app from your device's app store.",
    image: "/assets/step-1.png",
  },
  {
    number: 2,
    title: "Scan Your Space",
    description:
      "Walk around your home while the app analyzes WiFi signal strength.",
    image: "/assets/step-2.png",
  },
  {
    number: 3,
    title: "Optimize Placement",
    description:
      "Follow the app's guidance to place your WiFi extender for optimal coverage.",
    image: "/assets/step-3.png",
  },
];

export const BACKEND_URL =
  "https://frontiermap-backend-684147459334.us-central1.run.app/";
