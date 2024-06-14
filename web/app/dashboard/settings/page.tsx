"use client"
import CardComponent from "./card"
import { Title } from "@mantine/core";
export default function Demo() {
  return (
    <>
    <Title order={1} mt={20} mb={20}>
      Settings
    </Title>
       <CardComponent/>
     
    </>
  );
}