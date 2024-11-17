import { useQuery } from "@tanstack/react-query";
import { pinata } from "../../utils/config";

const fetchFileFromPinata = async () => {
  const response = await pinata.gateways.get(
    "bafkreiey3xkmc3jt6tbs5g4ufo3rdbtbugxfolhjxfh5hwx6knsfq37fja"
  );
  const blob = await response.data;
  if (blob instanceof Blob) {
    return blob;
  } else {
    throw new Error("Expected Blob but received: " + typeof blob);
  }
};

const usePinata = () => {
  const {
    data: fileData,
    error,
    isLoading,
  } = useQuery(["pinataFile"], fetchFileFromPinata);

  return { fileData, error, isLoading };
};

export default usePinata;
