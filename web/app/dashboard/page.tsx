"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  TextInput,
  ActionIcon,
  useMantineTheme,
  rem,
  Loader,
  Paper,
  Text,
  Avatar,
  Anchor,
  Group,
  Center,
  Space,
  Skeleton,
  Grid,
  Badge,
  List,
  Image,
  ThemeIcon,
  Title,
  Button,
  Modal

} from "@mantine/core";
import { useDisclosure } from '@mantine/hooks';
import { TableSelection } from "@/components/Table/Table";
import {
  IconSearch,
  IconArrowRight,
  IconFolderSearch,
  IconBrandNotion,
  IconMail,
  IconBrandAirtable,
  IconFile,
  IconWorldWww,
  IconExternalLink
} from "@tabler/icons-react";
import classes from "./global.module.css";
import Link from "next/link";
import NextImage from "next/image";
import { useRouter } from "next/navigation";

type ApiResponse = {
  answer: string;
  references: string[];
  similar_questions: string[];
};
type SearchResult = {
  id: string;
  page_link: string;
  title: string;
  snippet: string;
};


export default function InputWithButton(props: any) {
  const theme = useMantineTheme();
  const [opened, { open, close }] = useDisclosure(false);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [isSubmitted, setIsSubmitted] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(
    null
  );

  const [error, setError] = useState<string | null>(null);
  const [errorSecondAPI, setErrorSecondAPI] = useState<string | null>(null);

  const [query, setQuery] = useState<string>("");
  const [loadingFirstAPI, setLoadingFirstAPI] = useState<boolean>(false);
  const [loadingSecondAPI, setLoadingSecondAPI] = useState<boolean>(false);



  const handleFirstAPI = async (searchQuery: string) => {
    setLoadingFirstAPI(true);
    try {
      const response = await axios.post<ApiResponse>(
        "http://127.0.0.1:80/search/",
        { query: searchQuery }
      );
      console.log(response.data);
      console.log(response.data.similar_questions);
      console.log(response.data.answer);
      setData(response.data);
    } catch (error) {
      console.error("Error fetching data from first API:", error);
      setError(
        "Something went wrong with the first API. Please try again later."
      );
    }
    setLoadingFirstAPI(false);
  };

  const handleSecondAPI = async (searchQuery: string) => {
    setLoadingSecondAPI(true);
    try {
      const response = await axios.post<SearchResult[]>(
        "http://127.0.0.1:80/find/",
        { query: searchQuery }
      );
      console.log("Response from second API:", response.data);
      setSearchResults(response.data);
    } catch (error) {
      console.error("Error fetching data from second API:", error);
      setErrorSecondAPI(
        "Something went wrong with the second API. Please try again later."
      );
    }
    setLoadingSecondAPI(false);
  };

  const handleSearch = (event: React.FormEvent) => {
    event.preventDefault();
    setIsSubmitted(true);
    setError(null); // Reset the error state before making a new request
    setData(null); // Reset the data state to clear previous results

    handleFirstAPI(query);
    handleSecondAPI(query);

  };

  const handleRelatedSearchClick = async (relatedQuery: string) => {
    setData(null); // Reset the data state to clear previous results
    setQuery(relatedQuery); // Update the query state with the clicked related search
    handleFirstAPI(relatedQuery); // Fetch results for the clicked query
    handleSecondAPI(relatedQuery);
  };
  return (
    <>
     hey
    </>
  );
}
