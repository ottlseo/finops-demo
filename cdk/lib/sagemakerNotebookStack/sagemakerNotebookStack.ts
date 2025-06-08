import { Stack, StackProps, Fn } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';

export class SagemakerNotebookStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM Role
    const SageMakerNotebookinstanceRole = new iam.Role(this, 'SageMakerNotebookInstanceRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonOpenSearchServiceFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('SecretsManagerReadWrite')
      ],
    });
    
    // SageMaker Notebook Instance
    const cfnNotebookInstance = new sagemaker.CfnNotebookInstance(this, 'MyCfnNotebookInstance', {
      instanceType: 'ml.m5.xlarge',
      roleArn: SageMakerNotebookinstanceRole.roleArn,
      defaultCodeRepository: 'https://github.com/ottlseo/finops-demo.git',
      directInternetAccess: 'Enabled',
      notebookInstanceName: 'finops-notebook-instance',
      volumeSizeInGb: 50,
    });

    new cdk.CfnOutput(this, 'SagemakerNotebookInstance', {
      value: `${cfnNotebookInstance.notebookInstanceName}`,
      description: 'The name of SageMaker Notebook instance',
      exportName: 'SagemakerNotebookInstance',
    });
  }

}
