"use client"
import React from 'react';
import { Title, Text, Container, Space } from "@mantine/core";
import { ClusteringTable } from './table'; // Adjust the import path as needed

export default function Cluster() {
    return (
        <Container size="xl">
            <Title order={1} mt={20} mb={20}>
                Cluster
            </Title>
            <Text mb={30}>
                Fast, Simple, Scalable and accelerated GPU clusters focused on 
                simplicity, speed, & affordability.
            </Text>
            
            <Space h="md" />
            
            <Title order={2} mb={20}>
                Clustering Jobs
            </Title>
            <ClusteringTable />
        </Container>
    );
}