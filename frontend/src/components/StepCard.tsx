import React from "react";
import { Card, CardContent } from "./ui/card";
interface StepCardProps {
  number: number;
  title: string;
  description: string;
}

function StepCard({ number, title, description }: StepCardProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex flex-col items-center text-center space-y-4">
          <div className="flex items-center justify-center w-12 h-12 rounded-full bg-[#fe0036] text-white text-xl font-bold">
            {number}
          </div>
          <h3 className="text-xl font-bold">{title}</h3>
          <p className="text-gray-500 dark:text-gray-400">{description}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export default StepCard;
