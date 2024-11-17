import { Card, CardContent } from "./ui/card";

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <Card className="relative overflow-hidden group">
      <CardContent className="p-6">
        <div className="absolute inset-0 bg-gradient-to-r from-[#fe0036] to-[#ff6b8b] opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <div className="relative z-10 flex flex-col items-center text-center space-y-4">
          <div className="text-[#fe0036] group-hover:text-white transition-colors duration-300">
            {icon}
          </div>
          <h3 className="text-xl font-bold group-hover:text-white transition-colors duration-300">
            {title}
          </h3>
          <p className="text-gray-500 dark:text-gray-400 group-hover:text-white transition-colors duration-300">
            {description}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export default FeatureCard;
